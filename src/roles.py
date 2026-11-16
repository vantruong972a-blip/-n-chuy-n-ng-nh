"""Vai trò đo được của từng thuật toán + gợi ý mô hình theo phân khúc giá + khấu hao theo hãng của Linear Regression.

Mọi con số ở đây tính trên VALIDATION bằng mô hình fit CHỈ trên train (chưa thấy validation) => không dùng test để chọn.
Không có số liệu giả: nếu hai mô hình hơn kém nhau dưới ngưỡng thì khuyến nghị giữ mô hình mặc định.
"""
import numpy as np
import pandas as pd

import config as C
import evaluation as E
from model_registry import LABELS, ROLES

RECOMMEND_MARGIN = 0.03     # mô hình khác phải có RMSLE thấp hơn mô hình mặc định >= 3% (tương đối) mới được khuyến nghị
MIN_SEG = 150               # phân khúc ít hơn số này không đủ để kết luận


def _seg_stats(y, p):
    le = np.log(np.clip(p, 1.0, None) / y)
    return {"RMSLE": float(np.sqrt(np.mean(le ** 2))), "medAPE": float(np.median(np.abs(np.exp(le) - 1)) * 100),
            "bias_pct": float((np.exp(np.median(le)) - 1) * 100)}


def build_roles(X_val: pd.DataFrame, y_val: pd.Series, val_preds: dict, default_model: str) -> dict:
    """Trả {'table': DataFrame, 'recommend_by_band': list, 'roles': dict}."""
    y = np.asarray(y_val, float)
    key = np.asarray(val_preds[default_model], float)
    age = (C.REFERENCE_YEAR - X_val["year"]).to_numpy(float)
    km_known = (X_val["mileage_km"].fillna(0) > 0).to_numpy()
    segs = {}   # (loại, nhãn) -> mask
    edges = E.BAND_EDGES
    for lo, hi in zip(edges[:-1], edges[1:]):
        segs[("giá dự đoán", E._band_label(lo, hi))] = (key >= lo) & (key < hi)
    for lo, hi, lab in [(-1, 2, "0-2 năm"), (2, 5, "3-5 năm"), (5, 10, "6-10 năm"), (10, 20, "11-20 năm"), (20, 100, "trên 20 năm")]:
        segs[("tuổi xe", lab)] = (age > lo) & (age <= hi)
    segs[("số km", "có khai báo km")] = km_known
    segs[("số km", "không khai báo km")] = ~km_known
    segs[("toàn bộ", "toàn bộ validation")] = np.ones(len(y), bool)
    rows = []
    for (kind, lab), m in segs.items():
        if m.sum() < MIN_SEG and kind != "toàn bộ":
            continue
        for name, p in val_preds.items():
            rows.append({"model": name, "segment_type": kind, "segment": lab, "n": int(m.sum()), **_seg_stats(y[m], np.asarray(p, float)[m])})
    tab = pd.DataFrame(rows)

    # khuyến nghị theo dải giá dự đoán (merge dải nhỏ vào dải kề không cần thiết vì đã bỏ dải < MIN_SEG)
    rec, ordered = [], list(zip(edges[:-1], edges[1:]))
    for lo, hi in ordered:
        lab = E._band_label(lo, hi)
        t = tab[(tab.segment_type == "giá dự đoán") & (tab.segment == lab)]
        if t.empty:
            continue
        d = float(t.loc[t.model == default_model, "RMSLE"].iloc[0])
        best = t.loc[t.RMSLE.idxmin()]
        use = best.model if (best.model != default_model and best.RMSLE <= d * (1 - RECOMMEND_MARGIN)) else default_model
        rec.append({"lo": float(lo), "hi": None if hi == float("inf") else float(hi), "label": lab, "n": int(t.n.iloc[0]),
                    "model": use, "rmsle": {r.model: round(float(r.RMSLE), 4) for r in t.itertuples()}})
    # thế mạnh của từng mô hình: số phân khúc (không tính 'toàn bộ') mà mô hình có RMSLE / sai số trung vị thấp nhất
    inner = tab[tab.segment_type != "toàn bộ"]
    wins = {m: {"rmsle": 0, "medAPE": 0} for m in val_preds}
    for (_, _), t in inner.groupby(["segment_type", "segment"]):
        wins[t.loc[t.RMSLE.idxmin(), "model"]]["rmsle"] += 1
        wins[t.loc[t.medAPE.idxmin(), "model"]]["medAPE"] += 1
    n_seg = int(inner.groupby(["segment_type", "segment"]).ngroups)
    whole = tab[tab.segment_type == "toàn bộ"].set_index("model")
    roles = {}
    for m in val_preds:
        base = ROLES.get(m, {"title": LABELS.get(m, m), "summary": ""})
        roles[m] = {**base, "wins_rmsle": wins[m]["rmsle"], "wins_medAPE": wins[m]["medAPE"], "n_segments": n_seg,
                    "val_RMSLE": round(float(whole.loc[m, "RMSLE"]), 4), "val_medAPE": round(float(whole.loc[m, "medAPE"]), 2)}
    return {"table": tab, "recommend_by_band": rec, "roles": roles}


def roles_markdown(res: dict, default_model: str) -> str:
    L = ["# Vai trò đo được của từng thuật toán (tập validation, mô hình fit trên train)\n",
         "Sai số tương đối: RMSLE (log-giá, nhỏ hơn là tốt hơn) và sai số phần trăm trung vị (medAPE). 'Thắng' = thấp nhất trong phân khúc.\n",
         "| Mô hình | Vai trò | Thắng RMSLE | Thắng medAPE | RMSLE toàn bộ | medAPE toàn bộ (%) |", "|---|---|---|---|---|---|"]
    for m, r in res["roles"].items():
        L.append(f"| {LABELS.get(m, m)} | {r['title']} | {r['wins_rmsle']}/{r['n_segments']} | {r['wins_medAPE']}/{r['n_segments']} | {r['val_RMSLE']} | {r['val_medAPE']} |")
    L += ["", f"Mô hình mặc định: **{LABELS.get(default_model, default_model)}**. Khuyến nghị theo giá dự đoán (chỉ đổi khi tốt hơn >= {RECOMMEND_MARGIN:.0%}):\n",
          "| Giá dự đoán | n | Khuyến nghị | RMSLE từng mô hình |", "|---|---|---|---|"]
    for b in res["recommend_by_band"]:
        L.append(f"| {b['label']} | {b['n']} | {LABELS.get(b['model'], b['model'])} | " + ", ".join(f"{LABELS.get(k, k)} {v}" for k, v in b["rmsle"].items()) + " |")
    return "\n".join(L) + "\n"


def depreciation_by_brand(linear_pipe, X_train: pd.DataFrame, top=12, base_year=2017, base_km=40_000) -> pd.DataFrame:
    """Khấu hao THEO MÔ HÌNH Linear Regression cho dòng xe phổ biến nhất của từng hãng lớn: giá giảm bao nhiêu % khi xe già thêm 1 năm
    (giữ nguyên số km) và khi chạy thêm 10.000 km (giữ nguyên năm sản xuất). Đây là hệ quả của mô hình, không phải giá giao dịch."""
    rows = []
    for brand in X_train["brand"].value_counts().head(top).index:
        sub = X_train[X_train["brand"] == brand]
        model = sub["model"].value_counts().index[0]
        eng = sub.loc[sub["model"] == model, "engine_capacity"].median()
        base = {"brand": brand, "model": model, "year": base_year, "mileage_km": base_km, "engine_capacity": eng}
        older = {**base, "year": base_year - 1}
        farther = {**base, "mileage_km": base_km + 10_000}
        X = pd.DataFrame([base, older, farther]).reindex(columns=C.INPUT_FEATURES)
        p = linear_pipe.predict(X)
        rows.append({"brand": brand, "model": model, "base_price_vnd": float(p[0]),
                     "per_year_pct": float((p[1] / p[0] - 1) * 100), "per_10k_km_pct": float((p[2] / p[0] - 1) * 100)})
    return pd.DataFrame(rows)
