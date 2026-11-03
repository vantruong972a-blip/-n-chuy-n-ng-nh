"""Học có giám sát: Linear Regression, Random Forest, XGBoost trên giá xe cũ (VND) với train / validation / test theo NHÓM.

Quy trình mỗi tỷ lệ chia (seed cố định, nhóm tin rao trùng luôn nằm cùng một tập):
  1. Chia group-aware thành train / validation / test (test = test_size, validation = 25% của phần còn lại).
  2. Mỗi mô hình = Pipeline(feature engineering + tiền xử lý + mô hình, huấn luyện trên log1p(giá)).
     - BASELINE: tham số mặc định của thư viện, fit trên train.
     - TUNED: RandomizedSearchCV với GroupKFold CHỈ trên train (tiêu chí RMSE theo VND).
     - Cấu hình dùng cho mô hình = baseline hay tuned được chọn theo RMSE trên VALIDATION (không dùng test).
  3. Fit lại cấu hình đã chọn trên train+validation, rồi mới đo MAE/RMSE/R2/MAPE trên TEST (cùng một test cho cả 3 mô hình).
  4. Mô hình tốt nhất = RMSE validation thấp nhất. Test chỉ để báo cáo.
Ví dụ:
  python src/train_supervised.py                      # tỷ lệ phục vụ API (80/20), đầy đủ tinh chỉnh
  python src/train_supervised.py --all                # cả 4 tỷ lệ 60/40, 70/30, 80/20, 90/10 (lâu)
  python src/train_supervised.py --fast --max-rows 4000 --no-mlflow   # chạy thử nhanh
"""
import argparse
import json
import platform
import shutil
import time
from datetime import datetime, timezone

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, ParameterGrid, RandomizedSearchCV

import config as C
import evaluation as E
import monitoring
import tracking
from model_registry import LABELS, get_specs, make_pipeline
from preprocess import data_fingerprint, get_split3, load_clean

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = {"linear_regression": "#2a6f97", "random_forest": "#e08e0b", "xgboost": "#9a3b8f"}


def _strip(params):
    return {k.replace("model__regressor__", "").replace("pre__ct__cat__oh__", "onehot_"): v for k, v in params.items()}


def fit_model(est, X_tr, y_tr, g_tr, X_va, y_va, grid, tune, n_iter, cv, log_target=True):
    """Baseline + tuned trên train, so sánh trên validation. Trả về dict kết quả (chưa đụng tới test)."""
    t0 = time.perf_counter()
    base = make_pipeline(clone(est), log_target=log_target).fit(X_tr, y_tr)
    vb = E.metrics(y_va, base.predict(X_va))
    out = {"baseline_val": vb, "tuned_val": None, "cv_rmse": None, "cv_rmse_std": None, "best_params": {}}
    if tune and grid:
        k = min(n_iter, len(ParameterGrid(grid)))
        gs = RandomizedSearchCV(make_pipeline(clone(est), log_target=log_target), grid, n_iter=k, cv=cv,
                                scoring="neg_root_mean_squared_error", random_state=C.RANDOM_STATE, refit=True, n_jobs=1)
        gs.fit(X_tr, y_tr, groups=g_tr)
        out.update(cv_rmse=float(-gs.best_score_), cv_rmse_std=float(gs.cv_results_["std_test_score"][gs.best_index_]),
                   tuned_val=E.metrics(y_va, gs.best_estimator_.predict(X_va)), tuned_params=gs.best_params_)
        out["n_candidates"] = k
    use_tuned = out["tuned_val"] is not None and out["tuned_val"]["RMSE"] < vb["RMSE"]
    out["config"] = "tuned" if use_tuned else "baseline"
    out["params"] = out.get("tuned_params", {}) if use_tuned else {}
    out["val"] = out["tuned_val"] if use_tuned else vb
    out["fit_seconds"] = time.perf_counter() - t0
    return out


def run(test_sizes, model_names=None, tune=True, fast=False, use_mlflow=True, serve_size=C.SERVE_TEST_SIZE,
        max_rows=None, n_iter=None, ablate_target=True, promote=True):
    for d in (C.MODELS, C.REPORTS, C.PROCESSED):
        d.mkdir(parents=True, exist_ok=True)
    df = load_clean(max_rows=max_rows)
    data_version = data_fingerprint() + (f"-n{len(df)}" if max_rows else "")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_version = f"{stamp}-d{data_version[:6]}"
    specs = get_specs(fast=fast)
    if model_names:
        specs = {k: v for k, v in specs.items() if k in model_names}
    if not specs:
        raise SystemExit("Không có mô hình nào để huấn luyện.")
    n_iter = n_iter or (2 if fast else C.TUNE_ITER)
    mlflow_on = tracking.setup(use_mlflow)
    cv = GroupKFold(n_splits=C.CV_FOLDS)
    rows, served, ctx = [], {}, None

    for ts in test_sizes:
        pct = f"{round((1 - ts) * 100)}/{round(ts * 100)}"
        s = get_split3(ts, df=df)
        (X_tr, y_tr, g_tr), (X_va, y_va, _), (X_te, y_te, _) = s["train"], s["val"], s["test"]
        X_trva, y_trva = pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va])
        is_serve = abs(ts - serve_size) < 1e-9
        split_rows = []
        for name, (est, grid) in specs.items():
            r = fit_model(est, X_tr, y_tr, g_tr, X_va, y_va, grid, tune, n_iter, cv)
            final = make_pipeline(clone(est)).set_params(**r["params"]).fit(X_trva, y_trva)   # train+val, cấu hình đã chọn
            pred_te = final.predict(X_te)
            m = E.metrics(y_te, pred_te)                                 # test chỉ để báo cáo
            row = {"split": pct, "test_size": ts, "n_train": len(X_tr), "n_val": len(X_va), "n_test": len(X_te), "model": name,
                   "config_used": r["config"], "cv_rmse": r["cv_rmse"], "cv_rmse_std": r["cv_rmse_std"],
                   "val_RMSE_baseline": r["baseline_val"]["RMSE"],
                   "val_RMSE_tuned": None if r["tuned_val"] is None else r["tuned_val"]["RMSE"],
                   "val_RMSE": r["val"]["RMSE"], "val_MAE": r["val"]["MAE"], "val_R2": r["val"]["R2"],
                   **{k: m[k] for k in ("MAE", "RMSE", "R2", "MAPE", "medAPE", "within10pct", "within20pct")},
                   "train_seconds": r["fit_seconds"], "best_params": json.dumps(_strip(r["params"]), default=str),
                   "selected_best": False}
            split_rows.append((row, final, r, pred_te))
            print(f"[{pct}] {name:18s} val RMSE base={r['baseline_val']['RMSE']/1e6:7.1f}tr "
                  f"tuned={'-' if r['tuned_val'] is None else format(r['tuned_val']['RMSE']/1e6, '7.1f')}tr ({r['config']}) | "
                  f"TEST MAE={m['MAE']/1e6:6.1f}tr RMSE={m['RMSE']/1e6:7.1f}tr R2={m['R2']:.3f} medAPE={m['medAPE']:.1f}%", flush=True)
        winner = min(split_rows, key=lambda t: t[0]["val_RMSE"])
        winner[0]["selected_best"] = True
        for row, final, r, pred_te in split_rows:
            rows.append(row)
            with tracking.run(mlflow_on, f"{row['model']}_split{pct.replace('/', '-')}",
                              {"data_version": data_version, "model_version": model_version, "split": pct,
                               "selected_best": row["selected_best"], "served": is_serve, "config_used": row["config_used"]}):
                tracking.log(mlflow_on,
                             params={"algorithm": row["model"], "train_val_test": f"{len(X_tr)}/{len(X_va)}/{len(X_te)}",
                                     "test_size": ts, "split_strategy": "GroupShuffleSplit(group_id)", "cv": f"GroupKFold({cv.n_splits})",
                                     "tuning": f"RandomizedSearchCV(n_iter={n_iter})" if tune else "none", "target_transform": "log1p",
                                     "random_state": C.RANDOM_STATE, "data_version": data_version, "n_rows_clean": len(df),
                                     "features": ",".join(C.INPUT_FEATURES), "reference_year": C.REFERENCE_YEAR,
                                     **{f"hp_{k}": v for k, v in json.loads(row["best_params"]).items()}},
                             metrics={"cv_rmse": row["cv_rmse"], "val_RMSE_baseline": row["val_RMSE_baseline"],
                                      "val_RMSE_tuned": row["val_RMSE_tuned"], "val_RMSE": row["val_RMSE"], "val_R2": row["val_R2"],
                                      **{f"test_{k}": row[k] for k in ("MAE", "RMSE", "R2", "MAPE", "medAPE")},
                                      "train_seconds": row["train_seconds"]},
                             artifacts=[C.PROCESSED / "data_info.json", C.PROCESSED / "cleaning_report.json"],
                             model=final, model_name=row["model"])
            if is_serve:
                served[row["model"]] = (row, final, r, pred_te)
        if is_serve:
            ctx = {"X_tr": X_trva, "y_tr": y_trva, "X_va": X_va, "y_va": y_va, "X_te": X_te, "y_te": y_te, "split": s}

    res = pd.DataFrame(rows)
    res.to_csv(C.REPORTS / "supervised_results.csv", index=False)

    if served:
        best_name = min(served, key=lambda k: served[k][0]["val_RMSE"])
        _analysis(served, ctx, df, best_name, specs, ablate_target)
        info = _save_candidate(served, ctx, df, best_name, model_version, data_version, tune, n_iter, serve_size)
        ok, why = _check_candidate(info, ctx)
        if max_rows and C.MODELS == C.ROOT / "models":
            promote = False   # mẫu con chỉ để chạy thử, không bao giờ thay mô hình phục vụ
        if promote and ok:
            _promote(info["model_version"])
            print(f"\nĐã nâng cấp mô hình phục vụ API -> {info['model_version']} (mô hình tốt nhất: {best_name})")
        else:
            print(f"\nKHÔNG ghi đè models/ hiện tại. {'(--no-promote)' if ok else 'Lý do: ' + why}. "
                  f"Bản mới nằm ở {C.MODELS / 'versions' / info['model_version']}")
    _plots(res, served, ctx)
    print("\nĐã lưu:", C.REPORTS / "supervised_results.csv")
    return res


# --------------------------------------------------------------------------- phân tích + đóng gói
def _analysis(served, ctx, df, best_name, specs, ablate_target):
    X_te, y_te = ctx["X_te"], ctx["y_te"]
    te_full = df.loc[X_te.index]
    # 1) bảng so sánh (validation + test) + mức tham chiếu
    tab = pd.DataFrame([{"model": LABELS.get(k, k), "config": v[0]["config_used"], "val_RMSE_baseline": v[0]["val_RMSE_baseline"],
                         "val_RMSE_tuned": v[0]["val_RMSE_tuned"], "test_MAE": v[0]["MAE"], "test_RMSE": v[0]["RMSE"],
                         "test_R2": v[0]["R2"], "test_MAPE_%": v[0]["MAPE"], "test_medAPE_%": v[0]["medAPE"],
                         "test_within20%": v[0]["within20pct"], "selected_best": k == best_name} for k, v in served.items()])
    ref = E.metrics(y_te, np.full(len(y_te), float(ctx["y_tr"].median())))
    pd.DataFrame([{"reference": "luôn dự đoán trung vị giá train", **ref}]).to_csv(C.REPORTS / "reference_baseline.csv", index=False)
    tab.to_csv(C.REPORTS / "model_comparison_table.csv", index=False)
    f = lambda x: "-" if pd.isna(x) else f"{x / 1e6:,.1f}"  # noqa: E731
    md = ["| Mô hình | Cấu hình dùng | Val RMSE baseline (tr) | Val RMSE tuned (tr) | Test MAE (tr) | Test RMSE (tr) | Test R² | MAPE % | medAPE % | trong ±20% |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        md.append(f"| {r.model}{' ⭐' if r.selected_best else ''} | {r.config} | {f(r.val_RMSE_baseline)} | {f(r.val_RMSE_tuned)} | {f(r.test_MAE)} | "
                  f"{f(r.test_RMSE)} | {r.test_R2:.3f} | {r['test_MAPE_%']:.1f} | {r['test_medAPE_%']:.1f} | {r['test_within20%']:.0f}% |")
    md.append(f"| (tham chiếu: trung vị giá) | - | - | - | {f(ref['MAE'])} | {f(ref['RMSE'])} | {ref['R2']:.3f} | {ref['MAPE']:.1f} | {ref['medAPE']:.1f} | {ref['within20pct']:.0f}% |")
    (C.REPORTS / "model_comparison_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    # 2) permutation importance (cột đầu vào gốc), theo từng mô hình
    imps = []
    for k, v in served.items():
        t = E.permutation_table(v[1], X_te, y_te)
        t.insert(0, "model", k)
        imps.append(t)
    pd.concat(imps).to_csv(C.REPORTS / "feature_importance.csv", index=False)
    # 3) hiệu năng theo nhóm + ca sai nhiều nhất
    gp = []
    for k, v in served.items():
        g = E.group_performance(te_full, v[3])
        g.insert(0, "model", k)
        gp.append(g)
    pd.concat(gp).to_csv(C.REPORTS / "performance_by_group.csv", index=False)
    E.worst_errors(te_full, served[best_name][3]).to_csv(C.REPORTS / "worst_errors.csv", index=False)
    # 4) đối chứng: dự đoán trực tiếp giá (không log) vs log1p, tham số mặc định, trên VALIDATION
    if ablate_target:
        ab = []
        for k, (est, _) in specs.items():
            for lt in (True, False):
                p = make_pipeline(clone(est), log_target=lt).fit(ctx["X_tr"].loc[ctx["split"]["train"][0].index], ctx["split"]["train"][1])
                m = E.metrics(ctx["y_va"], p.predict(ctx["X_va"]))
                ab.append({"model": k, "target": "log1p(price)" if lt else "price (VND)", **m})
        pd.DataFrame(ab).to_csv(C.REPORTS / "target_transform_ablation.csv", index=False)


def _save_candidate(served, ctx, df, best_name, model_version, data_version, tune, n_iter, serve_size):
    vdir = C.MODELS / "versions" / model_version
    vdir.mkdir(parents=True, exist_ok=True)
    for name, (_, pipe, _, _) in served.items():
        joblib.dump(pipe, vdir / f"{name}.joblib")
    X_tr = ctx["X_tr"]
    brand_models = {b: g["model"].value_counts().index.tolist()[:200] for b, g in X_tr.groupby("brand")}
    ref = monitoring.build_reference(X_tr)
    ref["brand_models"] = brand_models
    (vdir / "reference_stats.json").write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
    cleaning = json.loads((C.PROCESSED / "data_info.json").read_text(encoding="utf-8")) if (C.PROCESSED / "data_info.json").exists() else {}
    info = {"model_version": model_version, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_version": data_version, "n_rows_clean": int(len(df)), "n_raw": cleaning.get("n_raw"),
            "split": f"{round((1 - serve_size) * 100)}/{round(serve_size * 100)}", "split_strategy": "GroupShuffleSplit theo group_id; train/val/test",
            "n_train": int(len(X_tr)), "n_val": int(len(ctx["X_va"])), "n_test": int(len(ctx["X_te"])), "target": C.TARGET,
            "target_transform": "log1p", "price_unit": C.PRICE_UNIT, "price_unit_note": C.PRICE_UNIT_NOTE,
            "reference_year": C.REFERENCE_YEAR, "features": C.INPUT_FEATURES, "required_features": C.REQUIRED_FEATURES,
            "random_state": C.RANDOM_STATE, "cv": f"GroupKFold({C.CV_FOLDS})", "tuned": tune, "tuning_iterations": n_iter,
            "selection_criterion": "RMSE (VND) thấp nhất trên tập validation; test chỉ để báo cáo", "best_model": best_name,
            "models": {k: {"label": LABELS.get(k, k), "config_used": v[0]["config_used"], "cv_rmse": v[0]["cv_rmse"],
                           "val_RMSE": v[0]["val_RMSE"],
                           **{m: v[0][m] for m in ("MAE", "RMSE", "R2", "MAPE", "medAPE", "train_seconds")},
                           "best_params": json.loads(v[0]["best_params"])} for k, v in served.items()},
            "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                          "numpy": np.__version__, "xgboost": _ver("xgboost"), "mlflow": _ver("mlflow")}}
    (vdir / "model_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def _check_candidate(info, ctx):
    """Cổng kiểm tra trước khi thay mô hình đang phục vụ: nạp được, dự đoán hữu hạn/dương, tốt hơn mức tham chiếu."""
    vdir = C.MODELS / "versions" / info["model_version"]
    try:
        X = ctx["X_te"].head(20)
        for name in info["models"]:
            p = joblib.load(vdir / f"{name}.joblib").predict(X)
            if not (np.isfinite(p).all() and (p > 0).all()):
                return False, f"{name} dự đoán giá không hữu hạn/không dương"
        best = info["models"][info["best_model"]]
        ref = E.metrics(ctx["y_te"], np.full(len(ctx["y_te"]), float(ctx["y_tr"].median())))
        if not (best["R2"] > 0 and best["RMSE"] < ref["RMSE"]):
            return False, "mô hình tốt nhất không tốt hơn mức tham chiếu (trung vị giá)"
    except Exception as e:  # noqa: BLE001
        return False, f"lỗi khi kiểm tra: {e}"
    return True, ""


def _promote(version):
    """Lưu bản đang phục vụ vào models/archive/<phiên bản> rồi mới chép bản mới lên models/."""
    src = C.MODELS / "versions" / version
    cur = C.MODELS / "model_info.json"
    if cur.exists():
        try:
            old = json.loads(cur.read_text(encoding="utf-8")).get("model_version", "previous")
        except json.JSONDecodeError:
            old = "previous"
        arch = C.MODELS / "archive" / old
        if not arch.exists():
            arch.mkdir(parents=True)
            for f in C.MODELS.iterdir():
                if f.is_file():
                    shutil.copy2(f, arch / f.name)
    for f in C.MODELS.glob("*.joblib"):
        f.unlink()   # tránh còn sót file của mô hình cũ (vd preprocessor.joblib) khác kiến trúc
    for f in src.iterdir():
        shutil.copy2(f, C.MODELS / f.name)


def _ver(pkg):
    try:
        return __import__(pkg).__version__
    except ImportError:
        return None


def _plots(res, served, ctx):
    order = [f"{round((1 - t) * 100)}/{round(t * 100)}" for t in C.TEST_SIZES]
    piv = res.pivot(index="split", columns="model", values="RMSE").reindex(order).dropna(how="all") / 1e6
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.8 / max(1, piv.shape[1])
    for i, m in enumerate(piv.columns):
        ax.bar(np.arange(len(piv)) + i * w, piv[m], w * 0.92, label=LABELS.get(m, m), color=PALETTE.get(m))
    ax.set_xticks(np.arange(len(piv)) + w * (piv.shape[1] - 1) / 2, piv.index)
    ax.set_ylabel("RMSE trên test (triệu VND)"); ax.set_xlabel("Tỷ lệ train+val / test")
    ax.set_title("So sánh mô hình theo tỷ lệ chia (thấp hơn là tốt hơn)"); ax.legend()
    fig.tight_layout(); fig.savefig(C.REPORTS / "model_comparison.png", dpi=150); plt.close(fig)
    if not served:
        return
    y = ctx["y_te"].to_numpy() / 1e6
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, (_, _, _, p) in served.items():
        axes[0].scatter(y, p / 1e6, s=6, alpha=.35, label=LABELS.get(name, name), color=PALETTE.get(name))
        axes[1].hist(np.clip((p - ctx["y_te"].to_numpy()) / ctx["y_te"].to_numpy() * 100, -100, 150), bins=60, alpha=.45,
                     label=LABELS.get(name, name), color=PALETTE.get(name))
    lo, hi = max(y.min(), 1), y.max()
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1); axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Giá niêm yết thực (triệu VND, log)"); axes[0].set_ylabel("Giá dự đoán (triệu VND, log)")
    axes[0].set_title("Dự đoán vs thực tế (test)"); axes[0].legend(markerscale=3)
    axes[1].set_xlabel("Sai số tương đối (%) (cắt ở -100..150)"); axes[1].set_title("Phân phối sai số tương đối (test)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(C.REPORTS / "prediction_errors.png", dpi=150); plt.close(fig)
    imp = pd.read_csv(C.REPORTS / "feature_importance.csv")
    fig, axes = plt.subplots(1, len(served), figsize=(5 * len(served), 4.6), sharey=False)
    for ax, (name, _) in zip(np.atleast_1d(axes), served.items()):
        t = imp[(imp.model == name) & (imp.kind == "column")].head(10).iloc[::-1]
        ax.barh(t.feature, t.rmse_increase_vnd / 1e6, xerr=t["std"] / 1e6, color=PALETTE.get(name))
        ax.set_title(LABELS.get(name, name)); ax.set_xlabel("RMSE tăng khi xáo cột (triệu VND)")
    fig.suptitle("Permutation importance trên test (10 cột đầu vào quan trọng nhất; xem thêm các nhóm cột trong feature_importance.csv)")
    fig.tight_layout(); fig.savefig(C.REPORTS / "feature_importance.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="chạy các tỷ lệ 60/40, 70/30, 80/20, 90/10")
    g.add_argument("--test-size", type=float, help="tỷ lệ test (vd 0.2 = 80/20)")
    ap.add_argument("--models", nargs="*", help="linear_regression random_forest xgboost")
    ap.add_argument("--no-tune", action="store_true", help="không tinh chỉnh siêu tham số (chỉ baseline)")
    ap.add_argument("--fast", action="store_true", help="lưới siêu tham số nhỏ (chạy thử nhanh)")
    ap.add_argument("--max-rows", type=int, help="chỉ dùng một mẫu theo nhóm (debug/kiểm thử)")
    ap.add_argument("--n-iter", type=int, help="số tổ hợp thử mỗi mô hình khi tinh chỉnh")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--no-promote", action="store_true", help="chỉ lưu vào models/versions/, không thay mô hình đang phục vụ")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    sizes = C.TEST_SIZES if a.all else [a.test_size or C.SERVE_TEST_SIZE]
    if C.SERVE_TEST_SIZE not in sizes and a.test_size:
        print(f"Lưu ý: tỷ lệ {a.test_size} khác tỷ lệ phục vụ API ({C.SERVE_TEST_SIZE}) nên sẽ không lưu mô hình cho API.")
    run(sizes, a.models, tune=not a.no_tune, fast=a.fast, use_mlflow=not a.no_mlflow, max_rows=a.max_rows,
        n_iter=a.n_iter, ablate_target=not a.no_ablation, promote=not a.no_promote)


if __name__ == "__main__":
    main()
