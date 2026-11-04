"""Gợi ý xe từ các tin rao THẬT đã làm sạch (Data/processed/clean.csv).

Hai cách, đúng như giao diện mô tả:
- content_based: lọc cứng theo mọi tiêu chí người dùng nhập, rồi xếp hạng theo độ phù hợp
  (khớp ngân sách, đời xe, số km, mức giá so với các xe cùng dòng và năm).
- knn: tìm k xe gần nhất (scikit-learn NearestNeighbors) với "xe lý tưởng" dựng từ tiêu chí.
  Chỉ lọc cứng ngân sách, hãng, dòng và năm; nhiên liệu/hộp số/kiểu dáng là tiêu chí mềm trong khoảng cách.

Không có ảnh và không có đường dẫn tin gốc trong dữ liệu, nên không trả ảnh. Mỗi xe chỉ xuất hiện một lần
cho mỗi group_id (các tin cùng một xe đăng lại không bị lặp).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

NUM_WEIGHTS = {"log_price": 2.0, "year": 1.5, "log_km": 1.0, "engine_capacity": 0.7, "seats": 0.5}
CAT_COLS = ["fuel_type", "transmission", "body_type", "origin"]
CAT_WEIGHT = 0.8
FUEL_VI = {"gasoline": "Xăng", "diesel": "Dầu diesel", "hybrid": "Hybrid", "electric": "Chạy điện"}
TRANS_VI = {"automatic": "Số tự động", "manual": "Số sàn", "semi_automatic": "Số bán tự động"}
ORIGIN_VI = {"domestic": "Lắp ráp trong nước", "imported": "Xe nhập khẩu"}
SOURCE_VI = {"bonbanh": "Bonbanh", "chotot": "Chợ Tốt"}


class DataNotAvailable(RuntimeError):
    pass


def _clean(v: Any) -> Any:
    if v is None or v is pd.NA:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


class Recommender:
    def __init__(self, df: pd.DataFrame):
        need = {"source", "source_id", "brand", "model", "year", "price", "group_id"}
        if not need <= set(df.columns):
            raise DataNotAvailable(f"Thiếu cột: {sorted(need - set(df.columns))}")
        d = df.copy().reset_index(drop=True)
        d = d[d["year"].notna() & d["price"].gt(0)].reset_index(drop=True)
        for c in ["brand", "model", "fuel_type", "transmission", "body_type", "origin"]:
            d[c] = d[c].astype(object).where(d[c].notna(), None).map(lambda x: x.lower().strip() if isinstance(x, str) else None)
        d["id"] = d["source"].astype(str) + "-" + d["source_id"].astype(str)
        d = d.drop_duplicates("id").reset_index(drop=True)
        self.df = d
        self.by_id = pd.Series(d.index.values, index=d["id"].values)
        # Trung vị giá theo (hãng, dòng, năm) chỉ khi có >= 5 tin, để nói "rẻ hơn mặt bằng" có căn cứ.
        g = d.groupby(["brand", "model", "year"])["price"]
        d["_grp_n"] = g.transform("size")
        med = g.transform("median")
        cv = g.transform("std") / med
        # Tên dòng trong dữ liệu đôi khi gộp nhiều mẫu khác nhau (vd "cx"); nhóm giá phân tán thì không so sánh.
        d["_grp_med"] = med.where((d["_grp_n"] >= 5) & (cv <= 0.3))
        # Chênh quá 25% so với trung vị nhiều khả năng do khác phiên bản hoặc tin nhập sai: không coi là "giá tốt".
        d.loc[(d["_grp_med"] - d["price"]) / d["_grp_med"] > 0.25, "_grp_med"] = np.nan
        # Ma trận đặc trưng cho KNN (chuẩn hóa trên toàn bộ dữ liệu).
        num = pd.DataFrame({
            "log_price": np.log(d["price"].astype(float)),
            "year": d["year"].astype(float),
            "log_km": np.log1p(d["mileage_km"].astype(float)),
            "engine_capacity": d["engine_capacity"].astype(float),
            "seats": d["seats"].astype(float),
        })
        self._med = num.median()
        self._mean = num.fillna(self._med).mean()
        self._std = num.fillna(self._med).std().replace(0, 1.0)
        self._cat_levels = {c: sorted(d[c].dropna().unique().tolist()) for c in CAT_COLS}
        self.X = self._encode_num(num)
        self.X_cat = self._encode_cat(d)
        self.X_all = np.hstack([self.X, self.X_cat])

    # ---------- mã hóa ----------
    def _encode_num(self, num: pd.DataFrame) -> np.ndarray:
        z = (num.fillna(self._med) - self._mean) / self._std
        return np.column_stack([z[c].to_numpy() * NUM_WEIGHTS[c] for c in NUM_WEIGHTS])

    def _encode_cat(self, d: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in CAT_COLS:
            for lv in self._cat_levels[c]:
                cols.append((d[c] == lv).fillna(False).to_numpy(dtype=float) * CAT_WEIGHT)
        return np.column_stack(cols) if cols else np.zeros((len(d), 0))

    # ---------- lọc ----------
    def _mask(self, f: Dict[str, Any], hard_attrs: bool) -> pd.Series:
        d = self.df
        m = pd.Series(True, index=d.index)
        if f.get("budget_min") is not None:
            m &= d["price"] >= f["budget_min"]
        if f.get("budget_max") is not None:
            m &= d["price"] <= f["budget_max"]
        if f.get("brand"):
            m &= d["brand"] == f["brand"].lower().strip()
        if f.get("model"):
            m &= d["model"] == f["model"].lower().strip()
        if f.get("year_from") is not None:
            m &= d["year"] >= f["year_from"]
        if hard_attrs:
            for k in CAT_COLS[:3]:
                if f.get(k):
                    m &= d[k] == f[k].lower().strip()
        return m

    # ---------- gợi ý ----------
    def recommend(self, f: Dict[str, Any], method: str = "content_based", limit: int = 12) -> Dict[str, Any]:
        d = self.df
        idx = np.flatnonzero(self._mask(f, hard_attrs=(method == "content_based")).to_numpy())
        total_groups = int(d.loc[idx, "group_id"].nunique()) if len(idx) else 0
        if len(idx) == 0:
            return {"items": [], "method": method, "total": 0}
        order = self._rank_knn(f, idx, limit) if method == "knn" else self._rank_content(f, idx)
        chosen, seen = [], set()
        for i in order:
            gid = d.at[i, "group_id"]
            if gid in seen:
                continue
            seen.add(gid)
            chosen.append(int(i))
            if len(chosen) >= limit:
                break
        return {"items": [self._listing(i) for i in chosen], "method": method, "total": total_groups}

    def _rank_content(self, f: Dict[str, Any], idx: np.ndarray) -> List[int]:
        d = self.df.loc[idx]
        parts, weights = [], []
        lo, hi = f.get("budget_min"), f.get("budget_max")
        if lo is not None and hi is not None and hi > lo:
            mid, half = (lo + hi) / 2, (hi - lo) / 2
            parts.append((1 - (d["price"] - mid).abs() / half).clip(0, 1)); weights.append(0.35)
        yr = d["year"].astype(float)
        parts.append(((yr - yr.min()) / (yr.max() - yr.min())) if yr.max() > yr.min() else pd.Series(0.5, index=d.index)); weights.append(0.30)
        km = d["mileage_km"].astype(float)
        rk = km.rank(pct=True, ascending=True)
        parts.append((1 - rk).fillna(0.5)); weights.append(0.20)
        val = ((d["_grp_med"] - d["price"]) / d["_grp_med"]).clip(-0.5, 0.5) + 0.5
        parts.append(val.fillna(0.5)); weights.append(0.15)
        score = sum(p * w for p, w in zip(parts, weights)) / sum(weights)
        return d.assign(_s=score).sort_values("_s", ascending=False, kind="stable").index.tolist()

    def _rank_knn(self, f: Dict[str, Any], idx: np.ndarray, limit: int) -> List[int]:
        d = self.df
        sub = d.loc[idx]
        lo, hi = f.get("budget_min"), f.get("budget_max")
        price = (lo + hi) / 2 if (lo is not None and hi is not None) else (hi if hi is not None else lo if lo is not None else sub["price"].median())
        q = pd.DataFrame({
            "log_price": [math.log(price)],
            "year": [float(f["year_from"]) if f.get("year_from") is not None else float(sub["year"].median())],
            "log_km": [np.nan], "engine_capacity": [np.nan], "seats": [np.nan],
        })
        qnum = self._encode_num(q)
        qcat = np.zeros((1, self.X_cat.shape[1]))
        pos = 0
        for c in CAT_COLS:
            for lv in self._cat_levels[c]:
                if f.get(c) and f[c].lower().strip() == lv:
                    qcat[0, pos] = CAT_WEIGHT
                pos += 1
        qv = np.hstack([qnum, qcat])
        k = min(len(idx), max(limit * 4, 20))
        nn = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(self.X_all[idx])
        _, nb = nn.kneighbors(qv)
        return [int(idx[j]) for j in nb[0]]

    def similar_to(self, car_id: str, k: int = 4) -> List[str]:
        """k xe gần nhất (KNN trên toàn bộ tin rao), bỏ chính xe này và các tin đăng lại của nó."""
        i = int(self.by_id[car_id])
        d = self.df
        nn = NearestNeighbors(n_neighbors=min(len(d), k * 6 + 1), algorithm="brute").fit(self.X_all)
        _, nb = nn.kneighbors(self.X_all[i:i + 1])
        gid0, out, seen = d.at[i, "group_id"], [], set()
        for j in nb[0]:
            gid = d.at[j, "group_id"]
            if j == i or gid == gid0 or gid in seen:
                continue
            seen.add(gid)
            out.append(d.at[j, "id"])
            if len(out) >= k:
                break
        return out

    # ---------- chi tiết ----------
    def get(self, car_id: str) -> Optional[Dict[str, Any]]:
        if car_id not in self.by_id.index:
            return None
        i = int(self.by_id[car_id])
        item = self._listing(i)
        item["similar_ids"] = self.similar_to(car_id)
        return item

    def _listing(self, i: int) -> Dict[str, Any]:
        r = self.df.loc[i]
        # Hộp số, nhiên liệu, số km đã có chip riêng trên thẻ xe; điểm nổi bật chỉ nêu thông tin bổ sung.
        hl: List[str] = []
        med = r["_grp_med"]
        if pd.notna(med) and r["price"] <= med * 0.95:
            pct = round((1 - r["price"] / med) * 100)
            hl.append(f"Giá thấp hơn khoảng {pct}% so với trung vị {int(r['_grp_n'])} tin cùng dòng và năm")
        km = _clean(r["mileage_km"])
        if km is not None and 0 < km <= 20000:
            hl.append("Đã đi ít (từ 20.000 km trở xuống)")
        if _clean(r["origin"]):
            hl.append(ORIGIN_VI.get(r["origin"], str(r["origin"])))
        if _clean(r["drivetrain"]) if "drivetrain" in r.index else False:
            hl.append(f"Dẫn động {str(r['drivetrain']).upper()}")
        return {
            "id": r["id"], "brand": _clean(r["brand"]), "model": _clean(r["model"]), "variant": _clean(r.get("variant")),
            "year": int(r["year"]), "mileage_km": _clean(r["mileage_km"]), "engine_capacity": _clean(r["engine_capacity"]),
            "fuel_type": _clean(r["fuel_type"]), "transmission": _clean(r["transmission"]), "body_type": _clean(r["body_type"]),
            "origin": _clean(r["origin"]), "province": _clean(r["province"]),
            "seats": None if _clean(r["seats"]) is None else int(r["seats"]),
            "price": int(r["price"]), "image_url": None, "highlights": hl[:4], "similar_ids": [],
            "source": SOURCE_VI.get(r["source"], str(r["source"])), "color": _clean(r.get("color")),
        }


_CACHE: Dict[str, Recommender] = {}


def load_recommender(path: Path) -> Recommender:
    key = str(path)
    if key not in _CACHE:
        if not Path(path).exists():
            raise DataNotAvailable("Chưa có dữ liệu tin rao đã làm sạch (Data/processed/clean.csv). Hãy chạy bước làm sạch dữ liệu trước.")
        _CACHE.clear()
        _CACHE[key] = Recommender(pd.read_csv(path, low_memory=False))
    return _CACHE[key]
