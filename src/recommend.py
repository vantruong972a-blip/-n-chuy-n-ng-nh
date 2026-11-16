"""Gợi ý xe từ các tin rao THẬT đã làm sạch (Data/processed/clean.csv).

Ba cách (method):
- content_based: Content-Based Filtering đúng nghĩa. Mỗi xe là một vector đặc trưng (hãng, dòng, nhiên liệu, hộp số, kiểu dáng,
  xuất xứ được mã hóa TF-IDF, nên dòng hiếm mang nhiều thông tin hơn dòng phổ biến; kèm giá, năm, km, dung tích, số chỗ).
  "Hồ sơ người dùng" dựng từ tiêu chí đã nhập và/hoặc các xe người dùng thích (liked_ids, trung bình vector các xe đó).
  Xe được xếp theo độ tương đồng cosine (phần phân loại) kết hợp độ gần của các thuộc tính số. Chỉ ngân sách, năm tối thiểu,
  hãng và dòng người dùng gõ là lọc cứng.
- rule_based: lọc cứng theo mọi tiêu chí, rồi chấm điểm theo luật (khớp ngân sách, đời xe, số km, giá so với mặt bằng).
- knn: tìm k xe gần nhất (scikit-learn NearestNeighbors) với "xe lý tưởng" dựng từ tiêu chí.
  Chỉ lọc cứng ngân sách, hãng, dòng và năm; nhiên liệu/hộp số/kiểu dáng là tiêu chí mềm trong khoảng cách.

Không có ảnh và không có đường dẫn tin gốc trong dữ liệu, nên không trả ảnh. Mỗi xe chỉ xuất hiện một lần
cho mỗi group_id (các tin cùng một xe đăng lại không bị lặp).
"""
from __future__ import annotations

import difflib
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

NUM_WEIGHTS = {"log_price": 2.0, "year": 1.5, "log_km": 1.0, "engine_capacity": 0.7, "seats": 0.5}
CAT_COLS = ["fuel_type", "transmission", "body_type", "origin"]
CAT_WEIGHT = 0.8
FUEL_VI = {"gasoline": "Xăng", "diesel": "Dầu diesel", "hybrid": "Hybrid", "electric": "Chạy điện"}
TRANS_VI = {"automatic": "Số tự động", "manual": "Số sàn", "semi_automatic": "Số bán tự động"}
ORIGIN_VI = {"domestic": "Lắp ráp trong nước", "imported": "Xe nhập khẩu"}
# Content-Based Filtering: trọng số từng trường phân loại, trọng số từng thuộc tính số và độ rộng hàm gần (đơn vị độ lệch chuẩn).
CB_FIELD_W = {"brand": 1.0, "model": 1.5, "fuel_type": 0.8, "transmission": 0.8, "body_type": 1.0, "origin": 0.5}
CB_NUM_W = {"log_price": 0.40, "year": 0.25, "log_km": 0.15, "engine_capacity": 0.10, "seats": 0.10}
CB_NUM_TAU = {"log_price": 0.6, "year": 0.8, "log_km": 1.0, "engine_capacity": 1.0, "seats": 1.0}
CB_CAT_SHARE = 0.6        # tỷ trọng phần phân loại khi hồ sơ có cả phần phân loại lẫn phần số
MAX_LIKED = 5
SOURCE_VI = {"bonbanh": "Bonbanh", "chotot": "Chợ Tốt"}


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
IMAGE_URL_PREFIX = "/car-images"


class DataNotAvailable(RuntimeError):
    pass


def norm_name(s: Any) -> str:
    """Chuẩn hóa tên hãng/dòng để so khớp: bỏ dấu, chữ thường, chỉ giữ chữ và số ("Vios G" -> "viosg")."""
    t = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "", t)


def resolve_names(query: str, names: Dict[str, int]) -> Dict[str, Any]:
    """Khớp tên người dùng gõ với tên có trong dữ liệu. `names` = {tên: số tin}.
    Thứ tự: khớp đúng -> khớp một phần (chứa nhau) -> gần đúng (difflib) -> không có (kèm gợi ý)."""
    q = norm_name(query)
    if not q:
        return {"kind": "none", "matched": [], "suggestions": []}
    by_norm: Dict[str, List[str]] = {}
    for n in names:
        by_norm.setdefault(norm_name(n), []).append(n)
    if q in by_norm:
        return {"kind": "exact", "matched": by_norm[q], "suggestions": []}
    part = [n for nn, lst in by_norm.items() for n in lst if (len(q) >= 2 and q in nn) or (len(nn) >= 3 and nn in q)]
    if part:
        part.sort(key=lambda n: -names[n])
        return {"kind": "partial", "matched": part[:8], "suggestions": []}
    close = difflib.get_close_matches(q, list(by_norm), n=3, cutoff=0.75)
    if close:
        return {"kind": "fuzzy", "matched": [n for c in close[:1] for n in by_norm[c]], "suggestions": []}
    sug = difflib.get_close_matches(q, list(by_norm), n=3, cutoff=0.6)
    return {"kind": "none", "matched": [], "suggestions": [n for c in sug for n in by_norm[c]][:3]}


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
    def __init__(self, df: pd.DataFrame, images_dir: Optional[Path] = None):
        need = {"source", "source_id", "brand", "model", "year", "price", "group_id"}
        if not need <= set(df.columns):
            raise DataNotAvailable(f"Thiếu cột: {sorted(need - set(df.columns))}")
        d = df.copy().reset_index(drop=True)
        d = d[d["year"].notna() & d["price"].gt(0)].reset_index(drop=True)
        for c in ["brand", "model", "fuel_type", "transmission", "body_type", "origin"]:
            d[c] = d[c].astype(object).where(d[c].notna(), None).map(lambda x: x.lower().strip() if isinstance(x, str) else None)
        d["id"] = d["source"].astype(str) + "-" + d["source_id"].astype(str)
        d = d.drop_duplicates("id").reset_index(drop=True)
        # Chữ ký để loại tin giống hệt nhau nhưng group_id khác (cùng xe đăng nhiều lần, khác tiêu đề).
        # Giá trị thiếu được thay bằng "?" trước khi nối chuỗi (nối với NaN cho NaN và sẽ gộp nhầm các tin khác nhau).
        def t(col: str) -> pd.Series:
            if col not in d.columns:
                return pd.Series("?", index=d.index)
            return d[col].astype(object).where(d[col].notna(), "?").astype(str)

        base = t("brand") + "|" + t("model") + "|" + t("year") + "|" + t("price") + "|" + t("transmission") + "|" + t("fuel_type")
        known = "km=" + t("mileage_km")
        unknown = "nokm|" + t("color") + "|" + t("variant")
        d["_sig"] = base + "|" + known.where(d["mileage_km"].notna(), unknown)
        self.df = d
        self.by_id = pd.Series(d.index.values, index=d["id"].values)
        self._brand_counts = d["brand"].value_counts().to_dict()
        self._model_counts = d.groupby("brand")["model"].value_counts().to_dict()
        self.images = self._load_images(images_dir)
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
        self._med = num.median().fillna(0.0)  # cột thiếu hoàn toàn (tập nhỏ) vẫn không sinh NaN
        self._mean = num.fillna(self._med).mean()
        self._std = num.fillna(self._med).std().fillna(1.0).replace(0, 1.0)
        self._cat_levels = {c: sorted(d[c].dropna().unique().tolist()) for c in CAT_COLS}
        self.X = self._encode_num(num)
        self.X_cat = self._encode_cat(d)
        self.X_all = np.hstack([self.X, self.X_cat])
        self._build_content_profile(num)

    @staticmethod
    def _load_images(images_dir: Optional[Path]) -> Dict[tuple, Dict[str, Any]]:
        """Đọc Data/images/CREDITS.csv; chỉ nhận ảnh có thật trên đĩa, đúng đuôi hợp lệ, không thoát khỏi thư mục ảnh."""
        out: Dict[tuple, Dict[str, Any]] = {}
        if not images_dir:
            return out
        root = Path(images_dir)
        f = root / "CREDITS.csv"
        if not f.exists():
            return out
        try:
            cr = pd.read_csv(f, dtype=str, keep_default_na=False)
        except Exception:
            return out
        for _, r in cr.iterrows():
            rel = str(r.get("file", "")).replace("\\", "/").strip("/")
            p = (root / rel).resolve()
            if not rel or ".." in rel.split("/") or not rel.lower().endswith(IMAGE_EXTS):
                continue
            if root.resolve() not in p.parents or not p.is_file():
                continue
            out[(str(r["brand"]).lower(), str(r["model"]).lower())] = {
                "url": f"{IMAGE_URL_PREFIX}/{rel}",
                "credit": {k: (r.get(k) or None) for k in ("title", "author", "license", "license_url", "source_url")},
            }
        return out

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

    # ---------- Content-Based Filtering ----------
    def _build_content_profile(self, num: pd.DataFrame) -> None:
        """Vector đặc trưng của từng xe: phần phân loại TF-IDF (thưa, chuẩn hóa L2) và phần số (z-score, giữ NaN)."""
        d, n = self.df, len(self.df)
        fields = {
            "brand": d["brand"],
            "model": (d["brand"] + " " + d["model"]).where(d["brand"].notna() & d["model"].notna()),
            "fuel_type": d["fuel_type"], "transmission": d["transmission"], "body_type": d["body_type"], "origin": d["origin"],
        }
        rows, cols, vals = [], [], []
        self._cb_token: Dict[tuple, tuple] = {}     # (trường, giá trị) -> (cột, trọng số idf * trọng số trường)
        base = 0
        for f, ser in fields.items():
            codes, uniques = pd.factorize(ser)
            ok = codes >= 0
            df_cnt = np.bincount(codes[ok], minlength=len(uniques))
            idf = np.log((n + 1) / (df_cnt + 1)) + 1.0
            w = idf * CB_FIELD_W[f]
            rows.append(np.flatnonzero(ok)); cols.append(base + codes[ok]); vals.append(w[codes[ok]])
            for j, u in enumerate(uniques):
                self._cb_token[(f, u)] = (base + j, float(w[j]))
            base += len(uniques)
        m = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, base))
        norm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
        self._cb_items = sparse.diags(np.divide(1.0, norm, out=np.zeros_like(norm), where=norm > 0)) @ m
        self._cb_dim = base
        self._cb_z = (num - self._mean) / self._std     # NaN giữ nguyên: thiếu dữ liệu thì độ gần trung tính

    def _cb_profile(self, f: Dict[str, Any], liked: List[int], idx: np.ndarray):
        """Hồ sơ người dùng: (vector phân loại đã chuẩn hóa | None, {thuộc tính số: giá trị z mục tiêu})."""
        vec = np.zeros(self._cb_dim)
        for fld in ("fuel_type", "transmission", "body_type"):
            if f.get(fld):
                tok = self._cb_token.get((fld, f[fld].lower().strip()))
                if tok:
                    vec[tok[0]] += tok[1]
        res = f.get("_resolved") or {}
        for b in res.get("brands") or []:
            tok = self._cb_token.get(("brand", b))
            if tok:
                vec[tok[0]] += tok[1] / len(res["brands"])
        if res.get("models"):
            toks = [self._cb_token[("model", f"{b} {m}")] for (b, m) in self._model_counts
                    if m in res["models"] and (res.get("brands") is None or b in res["brands"]) and ("model", f"{b} {m}") in self._cb_token]
            for col, w in toks:
                vec[col] += w / len(toks)
        nrm = np.linalg.norm(vec)
        explicit = vec / nrm if nrm > 0 else None
        liked_vec = None
        target: Dict[str, float] = {}
        if liked:
            lv = np.asarray(self._cb_items[liked].mean(axis=0)).ravel()
            ln = np.linalg.norm(lv)
            liked_vec = lv / ln if ln > 0 else None
            zl = self._cb_z.iloc[liked].mean()
            target = {c: float(zl[c]) for c in CB_NUM_W if pd.notna(zl[c])}
        parts = [v for v in (explicit, liked_vec) if v is not None]
        cat = None
        if parts:
            cat = np.sum(parts, axis=0)
            cat = cat / np.linalg.norm(cat)
        lo, hi = f.get("budget_min"), f.get("budget_max")
        if lo is not None and hi is not None:
            price = (lo + hi) / 2
        elif hi is not None:
            price = hi * 0.9
        elif lo is not None:
            price = lo * 1.1
        else:
            price = None
        if price is not None and price > 0:
            target["log_price"] = float((math.log(price) - self._mean["log_price"]) / self._std["log_price"])
        return cat, target

    def _rank_cbf(self, f: Dict[str, Any], idx: np.ndarray, liked: List[int]):
        cat, target = self._cb_profile(f, liked, idx)
        if cat is None and not target:
            return None, None
        sim_cat = (self._cb_items[idx] @ cat) if cat is not None else None
        sim_num = None
        if target:
            wsum = sum(CB_NUM_W[c] for c in target)
            sim_num = np.zeros(len(idx))
            for c, t in target.items():
                z = self._cb_z[c].to_numpy()[idx]
                k = np.exp(-0.5 * ((z - t) / CB_NUM_TAU[c]) ** 2)
                sim_num += CB_NUM_W[c] / wsum * np.where(np.isnan(z), 0.5, k)
        if sim_cat is not None and sim_num is not None:
            score = CB_CAT_SHARE * sim_cat + (1 - CB_CAT_SHARE) * sim_num
        else:
            score = sim_cat if sim_cat is not None else sim_num
        order = np.argsort(-score, kind="stable")
        return [int(idx[j]) for j in order], {int(idx[j]): float(score[j]) for j in order[:2000]}

    # ---------- lọc ----------
    def _resolve_filters(self, f: Dict[str, Any]) -> Dict[str, Any]:
        """Khớp mềm hãng/dòng. Trả {brands, models, notes, suggestions, empty}."""
        notes: List[str] = []
        sug: List[str] = []
        brands: Optional[List[str]] = None
        models: Optional[List[str]] = None
        if f.get("brand"):
            r = resolve_names(f["brand"], self._brand_counts)
            if r["kind"] == "none":
                return {"brands": [], "models": None, "notes": [f'Không có hãng "{f["brand"]}" trong dữ liệu.' + (f' Có thể bạn muốn: {", ".join(r["suggestions"])}.' if r["suggestions"] else "")], "suggestions": r["suggestions"], "suggestion_for": "brand", "empty": True}
            brands = r["matched"]
            if r["kind"] in ("partial", "fuzzy"):
                notes.append(f'Không có hãng đúng "{f["brand"]}"; đang hiển thị hãng gần đúng: {", ".join(brands)}.')
        if f.get("model"):
            pool: Dict[str, int] = {}
            for (b, m), n in self._model_counts.items():
                if brands is None or b in brands:
                    pool[m] = pool.get(m, 0) + int(n)
            r = resolve_names(f["model"], pool)
            if r["kind"] == "none":
                msg = f'Không có dòng xe "{f["model"]}"' + (f' của hãng {", ".join(brands)}' if brands else "") + " trong dữ liệu."
                if r["suggestions"]:
                    msg += f' Có thể bạn muốn: {", ".join(r["suggestions"])}.'
                return {"brands": brands, "models": [], "notes": notes + [msg], "suggestions": r["suggestions"], "suggestion_for": "model", "empty": True}
            models = r["matched"]
            if r["kind"] in ("partial", "fuzzy"):
                notes.append(f'Không có dòng đúng "{f["model"]}"; đang hiển thị dòng gần đúng: {", ".join(models)}.')
        return {"brands": brands, "models": models, "notes": notes, "suggestions": sug, "suggestion_for": None, "empty": False}

    def _mask(self, f: Dict[str, Any], hard_attrs: bool, resolved: Optional[Dict[str, Any]] = None) -> pd.Series:
        d = self.df
        res = resolved or self._resolve_filters(f)
        m = pd.Series(True, index=d.index)
        if f.get("budget_min") is not None:
            m &= d["price"] >= f["budget_min"]
        if f.get("budget_max") is not None:
            m &= d["price"] <= f["budget_max"]
        if res["brands"] is not None:
            m &= d["brand"].isin(res["brands"])
        if res["models"] is not None:
            m &= d["model"].isin(res["models"])
        if f.get("year_from") is not None:
            m &= d["year"] >= f["year_from"]
        if hard_attrs:
            for k in CAT_COLS[:3]:
                if f.get(k):
                    m &= d[k] == f[k].lower().strip()
        return m

    # ---------- gợi ý ----------
    def recommend(self, f: Dict[str, Any], method: str = "content_based", limit: int = 12, liked_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        d = self.df
        res = self._resolve_filters(f)
        notes = list(res["notes"])
        liked: List[int] = []
        for cid in (liked_ids or [])[:MAX_LIKED]:
            if cid in self.by_id.index:
                liked.append(int(self.by_id[cid]))
            else:
                notes.append(f"Không tìm thấy xe mã {cid} trong dữ liệu, đã bỏ qua.")
        out: Dict[str, Any] = {"items": [], "method": method, "total": 0, "notes": notes, "suggestions": res["suggestions"], "suggestion_for": res["suggestion_for"]}
        if res["empty"]:
            return out
        idx = np.flatnonzero(self._mask(f, hard_attrs=(method == "rule_based"), resolved=res).to_numpy())
        if method == "content_based" and liked:   # xe đã thích không tự gợi lại chính nó (hay các tin đăng lại của nó)
            ban_g, ban_s = {d.at[i, "group_id"] for i in liked}, {d.at[i, "_sig"] for i in liked}
            idx = idx[~(d.loc[idx, "group_id"].isin(ban_g) | d.loc[idx, "_sig"].isin(ban_s)).to_numpy()]
        if len(idx) == 0:
            return out
        sub = d.loc[idx]
        out["total"] = int(len(sub.drop_duplicates("group_id").drop_duplicates("_sig")))
        scores: Optional[Dict[int, float]] = None
        if method == "knn":
            order = self._rank_knn(f, idx, limit)
        elif method == "content_based":
            order, scores = self._rank_cbf({**f, "_resolved": res}, idx, liked)
            if order is None:
                order = self._rank_content(f, idx)
                out["notes"].append("Chưa có tiêu chí hay xe yêu thích để so khớp nên các xe được xếp theo độ mới, số km và mức giá.")
        else:
            order = self._rank_content(f, idx)
        chosen, seen_g, seen_s = [], set(), set()
        for i in order:
            gid, sig = d.at[i, "group_id"], d.at[i, "_sig"]
            if gid in seen_g or sig in seen_s:
                continue
            seen_g.add(gid)
            seen_s.add(sig)
            chosen.append(int(i))
            if len(chosen) >= limit:
                break
        out["items"] = [{**self._listing(i), **({"match_score": round(scores[i], 3)} if scores and i in scores else {})} for i in chosen]
        return out

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
        gid0, sig0, out, seen, seen_s = d.at[i, "group_id"], d.at[i, "_sig"], [], set(), set()
        for j in nb[0]:
            gid, sig = d.at[j, "group_id"], d.at[j, "_sig"]
            if j == i or gid == gid0 or sig == sig0 or gid in seen or sig in seen_s:
                continue
            seen.add(gid)
            seen_s.add(sig)
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
        ids = self.similar_to(car_id)
        item["similar_ids"] = ids
        item["similar"] = [self._listing(int(self.by_id[x])) for x in ids]   # đủ thông tin để hiện thẻ so sánh, không cần gọi thêm
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
            "price": int(r["price"]), "highlights": hl[:4], "similar_ids": [],
            "source": SOURCE_VI.get(r["source"], str(r["source"])), "color": _clean(r.get("color")),
            **self._image_fields(r["brand"], r["model"]),
        }

    def _image_fields(self, brand: Any, model: Any) -> Dict[str, Any]:
        """Ảnh minh họa theo (hãng, dòng) nếu có. Không phải ảnh của chính chiếc xe trong tin rao."""
        img = self.images.get((str(brand).lower(), str(model).lower()))
        if not img:
            return {"image_url": None, "image_kind": None, "image_credit": None}
        return {"image_url": img["url"], "image_kind": "model_illustration", "image_credit": img["credit"]}


_CACHE: Dict[str, Recommender] = {}


def load_recommender(path: Path, images_dir: Optional[Path] = None) -> Recommender:
    key = f"{path}|{images_dir}"
    if key not in _CACHE:
        if not Path(path).exists():
            raise DataNotAvailable("Chưa có dữ liệu tin rao đã làm sạch (Data/processed/clean.csv). Hãy chạy bước làm sạch dữ liệu trước.")
        _CACHE.clear()
        _CACHE[key] = Recommender(pd.read_csv(path, low_memory=False), images_dir)
    return _CACHE[key]
