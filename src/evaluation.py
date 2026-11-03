"""Chỉ số và phân tích đánh giá: bảng so sánh, permutation importance, hiệu năng theo nhóm xe, các ca sai nhiều nhất.

Mọi phân tích chạy trên tập TEST giữ riêng (hoặc validation khi ghi rõ) và chỉ để BÁO CÁO, không dùng để chọn mô hình.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config as C


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ape = np.abs(p - y) / y
    return {"MAE": float(mean_absolute_error(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p)), "MAPE": float(ape.mean() * 100), "medAPE": float(np.median(ape) * 100),
            "within10pct": float((ape <= 0.10).mean() * 100), "within20pct": float((ape <= 0.20).mean() * 100)}


FEATURE_GROUPS = {   # các cột tương quan mạnh: xáo riêng từng cột sẽ đánh giá thấp vai trò của nhóm => xáo cả nhóm cùng lúc
    "[nhóm] định danh xe (brand+model+variant+body_type)": ["brand", "model", "variant", "body_type"],
    "[nhóm] mức sử dụng (year+mileage_km)": ["year", "mileage_km"],
    "[nhóm] thông số máy (engine_capacity+transmission+fuel_type+drivetrain+seats)":
        ["engine_capacity", "transmission", "fuel_type", "drivetrain", "seats"],
    "[nhóm] thị trường (origin+province)": ["origin", "province"],
}


def permutation_table(pipe, X, y, n_repeats=3, max_rows=3000, seed=C.RANDOM_STATE) -> pd.DataFrame:
    """Permutation importance: tăng RMSE (VND) khi xáo trộn một cột đầu vào GỐC (hoặc cả nhóm cột tương quan),
    tính trên tập giữ riêng. Giá trị lớn = mô hình phụ thuộc nhiều. Cột tương quan chia sẻ vai trò nên xem cả các dòng [nhóm]."""
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
        y = y.loc[X.index]
    rng = np.random.RandomState(seed)
    y = np.asarray(y, float)
    base = float(np.sqrt(mean_squared_error(y, pipe.predict(X))))
    items = [(c, [c]) for c in X.columns] + [(k, v) for k, v in FEATURE_GROUPS.items()]
    rows = []
    for name, cols in items:
        inc = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            inc.append(float(np.sqrt(mean_squared_error(y, pipe.predict(Xp)))) - base)
        rows.append({"feature": name, "kind": "group" if name.startswith("[nhóm]") else "column",
                     "rmse_increase_vnd": float(np.mean(inc)), "std": float(np.std(inc))})
    return pd.DataFrame(rows).sort_values("rmse_increase_vnd", ascending=False).reset_index(drop=True)


def group_performance(df: pd.DataFrame, pred, min_n=30) -> pd.DataFrame:
    """df gồm các cột đầu vào + price (test). Trả bảng chỉ số theo từng nhóm xe (hãng, thân xe, tuổi, phân khúc giá, nguồn...)."""
    d = df.copy()
    d["pred"] = np.asarray(pred, float)
    d["age_band"] = pd.cut(C.REFERENCE_YEAR - d["year"], [-1, 2, 5, 10, 20, 100], labels=["0-2 năm", "3-5", "6-10", "11-20", ">20"])
    d["price_band"] = pd.cut(d[C.TARGET], [0, 3e8, 6e8, 1e9, 2e9, 1e12],
                             labels=["<300tr", "300-600tr", "0.6-1 tỷ", "1-2 tỷ", ">2 tỷ"])
    d["mileage_known"] = np.where(d["mileage_km"].fillna(0) > 0, "có km", "không có km")
    top_brands = d["brand"].value_counts().head(10).index
    d["brand_top10"] = d["brand"].where(d["brand"].isin(top_brands), "khác")
    rows = []
    for col in ["brand_top10", "body_type", "fuel_type", "transmission", "origin", "age_band", "price_band",
                "mileage_known", "source"]:
        if col not in d:
            continue
        for k, g in d.groupby(col, observed=True, dropna=False):
            if len(g) >= min_n:
                rows.append({"group_by": col, "group": str(k), "n": len(g), **metrics(g[C.TARGET], g["pred"])})
    return pd.DataFrame(rows)


def worst_errors(df: pd.DataFrame, pred, k=30) -> pd.DataFrame:
    """k ca sai tuyệt đối lớn nhất (VND) và k ca sai tương đối lớn nhất (%), kèm đặc trưng để tìm nguyên nhân."""
    d = df.copy()
    d["pred"] = np.asarray(pred, float)
    d["abs_err"] = (d["pred"] - d[C.TARGET]).abs()
    d["pct_err"] = (d["pred"] - d[C.TARGET]) / d[C.TARGET] * 100
    cols = ["source", "brand", "model", "variant", "year", "mileage_km", "engine_capacity", "province", C.TARGET, "pred",
            "abs_err", "pct_err"]
    a = d.nlargest(k, "abs_err")[[c for c in cols if c in d]].assign(rank_by="abs_err_vnd")
    b = d.assign(_a=d["pct_err"].abs()).nlargest(k, "_a")[[c for c in cols if c in d]].assign(rank_by="abs_pct_err")
    return pd.concat([a, b]).drop_duplicates(subset=["source", "brand", "model", "year", "mileage_km", C.TARGET]).reset_index()
