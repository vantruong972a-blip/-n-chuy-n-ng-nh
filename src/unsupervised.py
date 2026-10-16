"""Học không giám sát: PCA, K-Means, DBSCAN trên đặc trưng xe (KHÔNG dùng price).

- Chỉ fit bộ tiền xử lý, PCA, K-Means, DBSCAN trên tập TRAIN; tập test chỉ được chiếu/gán cụm
  (PCA.transform, KMeans.predict) để xem mô hình có khái quát hóa không. DBSCAN không có predict nên chỉ chạy trên train.
- price chỉ dùng SAU khi phân cụm để mô tả từng cụm. Phân cụm KHÔNG phải dự đoán giá.
"""
import json

import matplotlib
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

import config as C
from preprocess import build_preprocessor, get_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def choose_k(A, k_range=range(2, 9)):
    """Chọn k theo silhouette cao nhất (kèm inertia/Davies-Bouldin để tham khảo, vẽ ở k_selection.png)."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit(A)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(A, km.labels_),
                     "davies_bouldin": davies_bouldin_score(A, km.labels_)})
    tab = pd.DataFrame(rows)
    return int(tab.loc[tab.silhouette.idxmax(), "k"]), tab


def choose_dbscan(A, min_samples=5, max_noise=0.2):
    """Thử eps theo phân vị của khoảng cách k-láng giềng; chọn eps có silhouette cao nhất
    với >=2 cụm và nhiễu <= max_noise. Nếu không có eps hợp lệ -> dùng phân vị 90."""
    d = NearestNeighbors(n_neighbors=min_samples).fit(A).kneighbors(A)[0][:, -1]
    best = None
    for q in range(50, 100, 5):
        eps = float(np.percentile(d, q))
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(A)
        ncl, noise = len(set(lab)) - (1 if -1 in lab else 0), float((lab == -1).mean())
        if ncl >= 2 and noise <= max_noise:
            mask = lab != -1
            s = silhouette_score(A[mask], lab[mask])
            if best is None or s > best[0]:
                best = (s, eps)
    eps = best[1] if best else float(np.percentile(d, 90))
    return eps, min_samples, best is not None


def main():
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    X_tr, X_te, y_tr, y_te = get_split(C.SERVE_TEST_SIZE)
    pre = build_preprocessor(X_tr)
    A_tr, A_te = pre.fit_transform(X_tr), pre.transform(X_te)  # fit trên train

    pca = PCA(n_components=2, random_state=C.RANDOM_STATE).fit(A_tr)
    Z_tr, Z_te = pca.transform(A_tr), pca.transform(A_te)
    evr = pca.explained_variance_ratio_
    print("PCA explained variance:", evr.round(3), "tổng:", round(float(evr.sum()), 3))

    k, ktab = choose_k(A_tr)
    print(ktab.round(3).to_string(index=False), f"\n-> chọn k={k} (silhouette cao nhất)")
    km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit(A_tr)
    km_tr, km_te = km.labels_, km.predict(A_te)

    eps, ms, ok = choose_dbscan(A_tr)
    db = DBSCAN(eps=eps, min_samples=ms).fit_predict(A_tr)
    n_db, noise = len(set(db)) - (1 if -1 in db else 0), int((db == -1).sum())
    print(f"DBSCAN eps={eps:.2f} min_samples={ms} (chọn bằng lưới: {ok}) -> {n_db} cụm, {noise} điểm nhiễu")

    metrics = {"pca_explained_variance": evr.round(4).tolist(), "kmeans_k": k,
               "kmeans_silhouette_train": float(silhouette_score(A_tr, km_tr)),
               "kmeans_davies_bouldin_train": float(davies_bouldin_score(A_tr, km_tr)),
               "kmeans_silhouette_test_assigned": float(silhouette_score(A_te, km_te)) if len(set(km_te)) > 1 else None,
               "dbscan_eps": eps, "dbscan_min_samples": ms, "dbscan_n_clusters": n_db,
               "dbscan_noise_points": noise, "dbscan_noise_fraction": noise / len(db),
               "dbscan_silhouette_excl_noise": float(silhouette_score(A_tr[db != -1], db[db != -1])) if n_db >= 2 else None,
               "note": "Silhouette test chỉ để tham khảo khả năng khái quát; không dùng để chọn tham số."}
    (C.REPORTS / "unsupervised_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    ktab.to_csv(C.REPORTS / "kmeans_k_selection.csv", index=False)

    lab = pd.concat([
        pd.DataFrame({"row_id": X_tr.index, "set": "train", "kmeans": km_tr, "dbscan": db, "pc1": Z_tr[:, 0], "pc2": Z_tr[:, 1], "price": y_tr.values}),
        pd.DataFrame({"row_id": X_te.index, "set": "test", "kmeans": km_te, "dbscan": np.nan, "pc1": Z_te[:, 0], "pc2": Z_te[:, 1], "price": y_te.values})])
    lab.to_csv(C.REPORTS / "unsupervised_labels.csv", index=False)
    summ = lab[lab.set == "train"].groupby("kmeans")["price"].agg(["count", "mean", "median", "std"]).round(1)
    summ.to_csv(C.REPORTS / "cluster_price_summary.csv")
    print("\nGiá theo cụm K-Means (phân tích SAU phân cụm):\n", summ.to_string())

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].scatter(Z_tr[:, 0], Z_tr[:, 1], c=km_tr, cmap="viridis", s=18)
    ax[0].scatter(Z_te[:, 0], Z_te[:, 1], c=km_te, cmap="viridis", s=45, marker="x", label="test (gán cụm)")
    ax[0].set_title(f"K-Means (k={k}) trên PCA"); ax[0].legend()
    ax[1].scatter(Z_tr[:, 0], Z_tr[:, 1], c=db, cmap="tab10", s=18)
    ax[1].set_title(f"DBSCAN (eps={eps:.2f}): {n_db} cụm, {noise} nhiễu (nhãn -1)")
    ax[2].boxplot([y_tr.values[km_tr == c] for c in sorted(set(km_tr))], tick_labels=[f"cụm {c}" for c in sorted(set(km_tr))])
    ax[2].set_title("Giá theo cụm K-Means (chỉ để mô tả)"); ax[2].set_ylabel(f"price ({C.PRICE_UNIT})")
    for a in ax[:2]:
        a.set_xlabel(f"PC1 ({evr[0]:.0%})"); a.set_ylabel(f"PC2 ({evr[1]:.0%})")
    fig.tight_layout(); fig.savefig(C.REPORTS / "unsupervised.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(ktab.k, ktab.inertia, "o-"); ax[0].set_title("Inertia (elbow)")
    ax[1].plot(ktab.k, ktab.silhouette, "o-"); ax[1].set_title("Silhouette (càng cao càng tốt)")
    ax[2].plot(ktab.k, ktab.davies_bouldin, "o-"); ax[2].set_title("Davies-Bouldin (càng thấp càng tốt)")
    for a in ax:
        a.set_xlabel("k")
    fig.tight_layout(); fig.savefig(C.REPORTS / "k_selection.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
