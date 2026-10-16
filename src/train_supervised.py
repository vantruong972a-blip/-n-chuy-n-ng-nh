"""Học có giám sát: Linear Regression, Random Forest, XGBoost trên nhiều tỷ lệ train/test.

Quy trình mỗi tỷ lệ chia:
  1. Chia train/test (random_state cố định).
  2. Mỗi mô hình = Pipeline(tiền xử lý + mô hình); tinh chỉnh siêu tham số bằng GridSearchCV
     CHỈ trên tập train (KFold). Tiêu chí chọn: CV-RMSE thấp nhất.
  3. Chọn mô hình tốt nhất theo CV-RMSE (không nhìn test), rồi mới đo MAE/RMSE/R2 trên test.
Ví dụ:
  python src/train_supervised.py --all                # chạy cả 4 tỷ lệ
  python src/train_supervised.py --test-size 0.2      # chỉ 80/20
"""
import argparse
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline

import config as C
import monitoring
import tracking
from model_registry import LABELS, get_specs
from preprocess import build_preprocessor, data_fingerprint, get_split, load_clean, split_columns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _metrics(y, p):
    return {"MAE": float(mean_absolute_error(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p))}


def fit_one(name, est, grid, X_tr, y_tr, tune, cv):
    pipe = Pipeline([("pre", build_preprocessor(X_tr)), ("model", est)])
    t0 = time.perf_counter()
    if tune and grid:
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error", refit=True, n_jobs=1)
        gs.fit(X_tr, y_tr)
        best, params = gs.best_estimator_, gs.best_params_
        cv_rmse = float(-gs.best_score_)
        cv_std = float(gs.cv_results_["std_test_score"][gs.best_index_])
    else:
        sc = -cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error")
        cv_rmse, cv_std, params = float(sc.mean()), float(sc.std()), {}
        best = pipe.fit(X_tr, y_tr)
    return best, params, cv_rmse, cv_std, time.perf_counter() - t0


def run(test_sizes, model_names=None, tune=True, fast=False, use_mlflow=True, serve_size=C.SERVE_TEST_SIZE):
    for d in (C.MODELS, C.REPORTS, C.PROCESSED):
        d.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    version_data = data_fingerprint()
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_version = f"{run_stamp}-d{version_data[:6]}"
    specs = get_specs(fast=fast)
    if model_names:
        specs = {k: v for k, v in specs.items() if k in model_names}
    if not specs:
        raise SystemExit("Không có mô hình nào để huấn luyện.")
    mlflow_on = tracking.setup(use_mlflow)
    rows, served, ref = [], {}, None

    for ts in test_sizes:
        pct = f"{round((1 - ts) * 100)}/{round(ts * 100)}"
        X_tr, X_te, y_tr, y_te = get_split(ts, df=df)
        cv = KFold(n_splits=min(C.CV_FOLDS, len(X_tr)), shuffle=True, random_state=C.RANDOM_STATE)
        split_rows = []
        for name, (est, grid) in specs.items():
            best, params, cv_rmse, cv_std, secs = fit_one(name, est, grid, X_tr, y_tr, tune, cv)
            m = _metrics(y_te, best.predict(X_te))  # test chỉ dùng để báo cáo, không dùng để chọn
            row = {"split": pct, "test_size": ts, "n_train": len(X_tr), "n_test": len(X_te), "model": name,
                   "cv_rmse": cv_rmse, "cv_rmse_std": cv_std, **m, "train_seconds": secs,
                   "best_params": json.dumps({k.replace("model__", ""): v for k, v in params.items()}, default=str),
                   "selected_best": False}
            split_rows.append((row, best))
            print(f"[{pct}] {name:18s} CV-RMSE={cv_rmse:9.1f}  test: MAE={m['MAE']:8.1f} RMSE={m['RMSE']:8.1f} R2={m['R2']:.3f}")
        winner = min(split_rows, key=lambda t: t[0]["cv_rmse"])
        winner[0]["selected_best"] = True
        for row, best in split_rows:
            rows.append(row)
            is_serve = abs(ts - serve_size) < 1e-9
            with tracking.run(mlflow_on, f"{row['model']}_split{pct.replace('/', '-')}",
                              {"data_version": version_data, "model_version": model_version, "split": pct,
                               "selected_best": row["selected_best"], "served": is_serve}):
                tracking.log(mlflow_on,
                             params={"algorithm": row["model"], "train_test_ratio": pct, "test_size": ts,
                                     "n_train": row["n_train"], "n_test": row["n_test"], "cv_folds": cv.n_splits,
                                     "random_state": C.RANDOM_STATE, "n_features_raw": X_tr.shape[1],
                                     "data_version": version_data, **json.loads(row["best_params"])},
                             metrics={k: row[k] for k in ("cv_rmse", "MAE", "RMSE", "R2", "train_seconds")},
                             model=best, model_name=row["model"])
            if is_serve:
                served[row["model"]] = (row, best)
                if ref is None:
                    ref = (X_tr, y_te, X_te)

    res = pd.DataFrame(rows)
    res.to_csv(C.REPORTS / "supervised_results.csv", index=False)

    if served:  # lưu mô hình + thông tin phiên bản cho API (chỉ tỷ lệ serve_size)
        for name, (_, pipe) in served.items():
            joblib.dump(pipe, C.MODELS / f"{name}.joblib")
        best_name = min(served, key=lambda k: served[k][0]["cv_rmse"])
        X_tr_s = ref[0]
        num, cat = split_columns(X_tr_s)
        info = {"model_version": model_version, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "data_version": version_data, "n_rows_clean": len(df), "split": f"{round((1 - serve_size) * 100)}/{round(serve_size * 100)}",
                "n_train": int(len(X_tr_s)), "n_test": int(len(ref[2])), "target": C.TARGET,
                "price_unit": C.PRICE_UNIT, "price_unit_note": C.PRICE_UNIT_NOTE,
                "features": X_tr_s.columns.tolist(), "numeric_features": num, "categorical_features": cat,
                "random_state": C.RANDOM_STATE, "cv_folds": C.CV_FOLDS, "tuned": tune,
                "selection_criterion": "CV-RMSE thấp nhất trên tập train", "best_model": best_name,
                "models": {k: {"label": LABELS.get(k, k), **{m: v[0][m] for m in ("cv_rmse", "MAE", "RMSE", "R2", "train_seconds")},
                               "best_params": json.loads(v[0]["best_params"])} for k, v in served.items()},
                "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                              "pandas": pd.__version__, "numpy": np.__version__,
                              "xgboost": _ver("xgboost"), "mlflow": _ver("mlflow")}}
        (C.MODELS / "model_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        (C.MODELS / "reference_stats.json").write_text(
            json.dumps(monitoring.build_reference(X_tr_s), ensure_ascii=False), encoding="utf-8")
        _plots(res, served, ref, serve_size)
    else:
        _plots(res, {}, None, serve_size)
    print("\nĐã lưu:", C.REPORTS / "supervised_results.csv")
    if min(test_sizes) <= 0.1 + 1e-9:
        print("Lưu ý: tỷ lệ 90/10 cho tập test rất nhỏ nếu dataset nhỏ nên chỉ số dao động mạnh, không mặc định là tốt hơn.")
    return res


def _ver(pkg):
    try:
        return __import__(pkg).__version__
    except ImportError:
        return None


def _plots(res, served, ref, serve_size):
    piv = res.pivot(index="split", columns="model", values="RMSE").reindex(
        [f"{round((1 - t) * 100)}/{round(t * 100)}" for t in C.TEST_SIZES]).dropna(how="all")
    ax = piv.plot(kind="bar", figsize=(8, 4.5), rot=0)
    ax.set_ylabel("RMSE trên test"); ax.set_xlabel("Tỷ lệ train/test"); ax.set_title("So sánh mô hình theo tỷ lệ chia")
    plt.tight_layout(); plt.savefig(C.REPORTS / "model_comparison.png", dpi=150); plt.close()
    if served and ref:
        _, y_te, X_te = ref
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for name, (_, pipe) in served.items():
            p = pipe.predict(X_te)
            axes[0].scatter(y_te, p, s=18, alpha=.7, label=LABELS.get(name, name))
            axes[1].hist(p - y_te, bins=15, alpha=.5, label=LABELS.get(name, name))
        lo, hi = float(y_te.min()), float(y_te.max())
        axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
        axes[0].set_xlabel("Giá thực"); axes[0].set_ylabel("Giá dự đoán"); axes[0].set_title("Dự đoán vs thực tế (test)"); axes[0].legend()
        axes[1].set_xlabel("Sai số (dự đoán - thực)"); axes[1].set_title("Phân phối sai số (test)"); axes[1].legend()
        plt.tight_layout(); plt.savefig(C.REPORTS / "prediction_errors.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="chạy các tỷ lệ 60/40, 70/30, 80/20, 90/10")
    g.add_argument("--test-size", type=float, help="tỷ lệ test (vd 0.2 = 80/20)")
    ap.add_argument("--models", nargs="*", help="linear_regression random_forest xgboost")
    ap.add_argument("--no-tune", action="store_true", help="không tinh chỉnh siêu tham số")
    ap.add_argument("--fast", action="store_true", help="lưới siêu tham số nhỏ (chạy thử nhanh)")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    sizes = C.TEST_SIZES if a.all else [a.test_size or C.SERVE_TEST_SIZE]
    run(sizes, a.models, tune=not a.no_tune, fast=a.fast, use_mlflow=not a.no_mlflow)


if __name__ == "__main__":
    main()
