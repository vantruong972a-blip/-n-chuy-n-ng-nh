"""Chỉ số và phân tích đánh giá: bảng so sánh, permutation importance, hiệu năng theo nhóm xe, các ca sai nhiều nhất.

Mọi phân tích chạy trên tập TEST giữ riêng (hoặc validation khi ghi rõ) và chỉ để BÁO CÁO, không dùng để chọn mô hình.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config as C


def rmsle_score(y, p) -> float:
    """RMSE trên log-giá (RMSLE). Nhỏ hơn = sai số TƯƠNG ĐỐI nhỏ hơn. Dùng làm tiêu chí chọn mô hình/siêu tham số vì RMSE theo VND
    bị chi phối bởi ~6% xe trên 2 tỷ (chiếm ~95% tổng bình phương sai số) nên rất nhiễu giữa các lần chia dữ liệu."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean(np.log(np.clip(p, 1.0, None) / y) ** 2)))


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ape = np.abs(p - y) / y
    rmsle = float(np.sqrt(np.mean(np.log(np.clip(p, 1.0, None) / y) ** 2)))   # sai số tương đối (log): không bị vài xe siêu sang chi phối
    return {"RMSLE": rmsle, "MAE": float(mean_absolute_error(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
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


# ----------------------------------------------------------------------------- khoảng sai số theo mức giá
BAND_EDGES = [0, 300e6, 500e6, 800e6, 1.2e9, 2e9, float("inf")]   # theo GIÁ DỰ ĐOÁN (VND), biết được lúc suy luận
BAND_LEVEL = 0.8                                                  # khoảng chứa ~80% giá thật
BAND_MIN_N = 100                                                  # dải ít hơn mức này sẽ được gộp với dải kề


def _band_label(lo: float, hi: float) -> str:
    f = lambda v: f"{v / 1e9:g} tỷ" if v >= 1e9 else f"{v / 1e6:g} triệu"
    if lo <= 0:
        return f"dưới {f(hi)}"
    return f"trên {f(lo)}" if hi == float("inf") else f"{f(lo)} – {f(hi)}"


def error_bands(y_true, y_pred, edges=BAND_EDGES, level=BAND_LEVEL, min_n=BAND_MIN_N) -> list:
    """Phân vị của tỷ lệ giá thật / giá dự đoán theo dải giá dự đoán, tính trên tập test (mô hình chưa từng thấy).

    Mỗi dải: {lo, hi, label, n, q_low, q_high} với q_* là hệ số nhân (giá thật ≈ dự đoán × q).
    Dải có ít hơn min_n mẫu được gộp với dải kề để phân vị đủ ổn định.
    """
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0) & (p > 0)
    y, p = y[ok], p[ok]
    ratio = np.log(y / p)
    idx = np.searchsorted(np.asarray(edges[1:-1], float), p, side="right")
    groups = [[i] for i in range(len(edges) - 1)]
    cnt = lambda g: int(np.isin(idx, g).sum())
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for k, g in enumerate(groups):
            if cnt(g) < min_n:
                j = k - 1 if k > 0 else k + 1
                if k > 0 and k + 1 < len(groups) and cnt(groups[k + 1]) < cnt(groups[k - 1]):
                    j = k + 1
                lo_k, hi_k = sorted((k, j))
                groups[lo_k:hi_k + 1] = [groups[lo_k] + groups[hi_k]]
                changed = True
                break
    a = (1 - level) / 2
    out = []
    for g in groups:
        m = np.isin(idx, g)
        if m.sum() < 20:
            continue
        lo, hi = edges[min(g)], edges[max(g) + 1]
        ql, qh = np.quantile(ratio[m], [a, 1 - a])
        out.append({"lo": float(lo), "hi": None if hi == float("inf") else float(hi), "label": _band_label(lo, hi),
                    "n": int(m.sum()), "q_low": float(np.exp(ql)), "q_high": float(np.exp(qh)),
                    "median_ratio": float(np.exp(np.median(ratio[m])))})
    return out


def apply_band(bands: list, pred: float, level: float = BAND_LEVEL):
    """Khoảng [thấp, cao] cho một giá dự đoán; None nếu không có thông tin dải."""
    for b in bands or []:
        if pred >= b["lo"] and (b["hi"] is None or pred < b["hi"]):
            return {"low": pred * b["q_low"], "high": pred * b["q_high"], "level": level, "band": b["label"], "n": b["n"]}
    return None
