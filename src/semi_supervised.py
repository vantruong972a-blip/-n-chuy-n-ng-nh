"""Học bán giám sát: Self-Training / Pseudo-Labeling cho hồi quy (Random Forest).

Thí nghiệm có kiểm soát:
  - Chia train/test như bình thường. Tập TEST giữ nguyên nhãn thật, chỉ dùng để đánh giá cuối.
  - Ẩn ngẫu nhiên một tỷ lệ nhãn `price` của tập TRAIN (--hide), coi đó là dữ liệu không nhãn.
  - Baseline: mô hình chỉ dùng phần train CÓ nhãn.  Self-training: thêm pseudo-label tin cậy vào tập huấn luyện.
  - Oracle (tham khảo): mô hình dùng toàn bộ nhãn train (cho thấy "trần" nếu không ẩn nhãn).
Tiêu chí tin cậy (hồi quy): độ lệch chuẩn dự đoán giữa các cây của RF; mỗi vòng chỉ nhận keep_frac
mẫu không nhãn có độ bất định thấp nhất. Nhãn thật bị ẩn KHÔNG được dùng để huấn luyện; chỉ dùng để
chẩn đoán chất lượng pseudo-label (cột pseudo_mae_vs_hidden_true).
Hạn chế: pseudo-label có thể lặp lại và khuếch đại sai số của mô hình ban đầu, nên kết quả có thể XẤU HƠN baseline.
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config as C
from preprocess import build_preprocessor, get_split, load_clean

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _rf(n_trees):
    return RandomForestRegressor(n_estimators=n_trees, random_state=C.RANDOM_STATE, n_jobs=-1)


def _ev(m, A, y):
    p = m.predict(A)
    return {"MAE": float(mean_absolute_error(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))), "R2": float(r2_score(y, p))}


def experiment(A_tr, y_tr, A_te, y_te, A_ext, hide, seed, rounds, keep_frac, pseudo_weight, n_trees):
    rng = np.random.RandomState(seed)
    n = len(y_tr)
    hidden = np.zeros(n, bool)
    hidden[rng.choice(n, int(round(n * hide)), replace=False)] = True
    A_l, y_l = A_tr[~hidden], y_tr[~hidden]
    A_pool = A_tr[hidden] if A_ext is None else np.vstack([A_tr[hidden], A_ext])
    y_true_pool = np.r_[y_tr[hidden], np.full(0 if A_ext is None else len(A_ext), np.nan)]  # chỉ để chẩn đoán

    base = _rf(n_trees).fit(A_l, y_l)
    r_base = _ev(base, A_te, y_te)
    r_orc = _ev(_rf(n_trees).fit(A_tr, y_tr), A_te, y_te)

    Xc, yc, w = A_l.copy(), y_l.copy(), np.ones(len(y_l))
    pool, ytp = A_pool.copy(), y_true_pool.copy()
    pseudo_err, n_pseudo = [], 0
    for _ in range(rounds):
        if len(pool) == 0:
            break
        m = _rf(n_trees).fit(Xc, yc, sample_weight=w)
        preds = np.stack([t.predict(pool) for t in m.estimators_])
        mean, std = preds.mean(0), preds.std(0)
        pick = np.argsort(std)[: max(1, int(len(pool) * keep_frac))]
        Xc, yc = np.vstack([Xc, pool[pick]]), np.r_[yc, mean[pick]]
        w = np.r_[w, np.full(len(pick), pseudo_weight)]
        ok = ~np.isnan(ytp[pick])
        if ok.any():
            pseudo_err.extend(np.abs(mean[pick][ok] - ytp[pick][ok]))
        n_pseudo += len(pick)
        keep = np.setdiff1d(np.arange(len(pool)), pick)
        pool, ytp = pool[keep], ytp[keep]
    final = _rf(n_trees).fit(Xc, yc, sample_weight=w)
    r_st = _ev(final, A_te, y_te)
    return {"hide_frac": hide, "seed": seed, "n_labeled": int((~hidden).sum()), "n_unlabeled_pool": len(A_pool),
            "n_pseudo_used": n_pseudo, "pseudo_mae_vs_hidden_true": float(np.mean(pseudo_err)) if pseudo_err else np.nan,
            **{f"baseline_{k}": v for k, v in r_base.items()}, **{f"selftrain_{k}": v for k, v in r_st.items()},
            **{f"oracle_{k}": v for k, v in r_orc.items()},
            "delta_RMSE_selftrain_minus_baseline": r_st["RMSE"] - r_base["RMSE"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hide", nargs="*", type=float, default=[0.3, 0.5, 0.7, 0.9], help="tỷ lệ nhãn train bị ẩn")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--keep-frac", type=float, default=0.5)
    ap.add_argument("--pseudo-weight", type=float, default=0.5)
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--use-external", action="store_true", help="thêm Data/external/car_unlabeled.csv vào pool không nhãn")
    a = ap.parse_args()
    C.REPORTS.mkdir(parents=True, exist_ok=True)

    X_tr, X_te, y_tr, y_te = get_split(C.SERVE_TEST_SIZE)
    pre = build_preprocessor(X_tr)
    A_tr, A_te = pre.fit_transform(X_tr), pre.transform(X_te)  # chỉ fit trên đặc trưng train
    A_ext = None
    if a.use_external:
        if C.UNLABELED_CSV.exists():
            ext = load_clean(C.UNLABELED_CSV, with_target=False).drop(columns=[C.TARGET], errors="ignore")
            A_ext = pre.transform(ext[X_tr.columns])
        else:
            print("Không thấy car_unlabeled.csv, bỏ qua --use-external")
    rows = []
    for h in a.hide:
        for s in range(a.seeds):
            rows.append(experiment(A_tr, y_tr.values, A_te, y_te.values, A_ext, h, s, a.rounds, a.keep_frac, a.pseudo_weight, a.trees))
        print(f"hide={h:.0%} xong")
    res = pd.DataFrame(rows)
    res.to_csv(C.REPORTS / "semi_supervised_results.csv", index=False)
    summ = res.groupby("hide_frac").agg(n_labeled=("n_labeled", "first"), baseline_RMSE=("baseline_RMSE", "mean"),
                                        selftrain_RMSE=("selftrain_RMSE", "mean"), oracle_RMSE=("oracle_RMSE", "mean"),
                                        delta_mean=("delta_RMSE_selftrain_minus_baseline", "mean"),
                                        delta_std=("delta_RMSE_selftrain_minus_baseline", "std"),
                                        n_better=("delta_RMSE_selftrain_minus_baseline", lambda d: int((d < 0).sum()))).round(2)
    summ["n_runs"] = a.seeds
    summ.to_csv(C.REPORTS / "semi_supervised_summary.csv")
    print("\nTóm tắt (RMSE trên test có nhãn thật; delta<0 nghĩa là self-training tốt hơn baseline):\n", summ.to_string())

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(summ))
    ax.bar(x - .27, summ.baseline_RMSE, .27, label="Baseline (chỉ nhãn có)")
    ax.bar(x, summ.selftrain_RMSE, .27, label="Self-training")
    ax.bar(x + .27, summ.oracle_RMSE, .27, label="Oracle (đủ nhãn)")
    ax.set_xticks(x, [f"ẩn {h:.0%}" for h in summ.index]); ax.set_ylabel("RMSE test (trung bình các seed)")
    ax.set_title("Bán giám sát vs baseline"); ax.legend()
    fig.tight_layout(); fig.savefig(C.REPORTS / "semi_supervised.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
