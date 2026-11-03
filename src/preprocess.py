"""Làm sạch dữ liệu thật, tạo tập có nhãn / chưa nhãn, và chia train/validation/test theo NHÓM tin rao.

Nguyên tắc:
- Data/raw/ không bao giờ bị sửa. Kết quả ghi ra Data/processed/ (clean.csv, unlabeled.csv, rejected_rows.csv,
  cleaning_report.json, data_info.json).
- Chỉ loại dòng khi có BẰNG CHỨNG là dữ liệu lỗi (quy tắc cụ thể, ghi lý do từng dòng vào rejected_rows.csv).
  Xe sang/xe cổ giá cao nhưng nhất quán với dòng xe của nó được GIỮ LẠI.
- Chia dữ liệu theo group_id (các tin rao trùng/gần trùng của cùng một xe luôn nằm cùng một tập) để không rò rỉ.
- Bộ tiền xử lý (features.build_preprocessor) chỉ fit trên train, nằm trong Pipeline cùng mô hình.
"""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config as C
import data_sources
from features import build_preprocessor, clean_mileage  # noqa: F401  (build_preprocessor re-export cho các module cũ)


# ----------------------------------------------------------------------------- phiên bản dữ liệu
def data_fingerprint() -> str:
    """Mã phiên bản dữ liệu = SHA-256 rút gọn của các file thô hiện có (đổi dữ liệu => đổi mã)."""
    h = hashlib.sha256()
    for name in sorted(C.RAW_FILES):
        p = C.RAW_FILES[name]
        if p.exists():
            h.update(name.encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:12]


# ----------------------------------------------------------------------------- làm sạch
def _reject(df, mask, reason, rejected):
    if mask.any():
        r = df.loc[mask, ["source", "source_id", "brand", "model", "year", "mileage_km", "price"]].copy()
        r["reason"] = reason
        rejected.append(r)
    return df.loc[~mask]


def build_dataset(raw: pd.DataFrame | None = None):
    """Trả về (labeled, unlabeled, rejected, report). `raw` mặc định = toàn bộ Data/raw/."""
    raw = data_sources.load_all() if raw is None else raw.copy()
    rep = {"n_raw": int(len(raw)), "n_raw_by_source": raw.groupby("source").size().to_dict(), "steps": []}
    rejected: list[pd.DataFrame] = []

    def step(name, df_before, df_after, note=""):
        rep["steps"].append({"step": name, "removed": int(len(df_before) - len(df_after)), "remaining": int(len(df_after)),
                             "note": note})

    df = raw.reset_index(drop=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # 1) Trùng lặp: cùng một tin (source, source_id) bị thu thập 2 lần, hoặc tin đăng lại y hệt mọi thuộc tính (kể cả giá)
    cols = [c for c in data_sources.CANON_COLS if c != "source_id"]
    dup = df.duplicated(subset=["source", "source_id"], keep="first") | df.duplicated(subset=cols, keep="first")
    df = _reject(df, dup, "duplicate_listing", rejected)
    step("loai_tin_trung_lap", raw, df, "cùng source_id hoặc giống hệt mọi thuộc tính (kể cả giá)")

    # 2) Quy tắc có bằng chứng (thứ tự cố định)
    b = df
    df = _reject(df, df["price"].isna() | (df["price"] <= 0), "price_missing_or_nonpositive", rejected)
    step("gia_thieu_hoac_khong_duong", b, df)

    b = df
    bad_year = df["year"].isna() | (df["year"] < C.YEAR_MIN) | (df["year"] > C.YEAR_MAX)
    df = _reject(df, bad_year, "year_invalid", rejected)
    step("nam_san_xuat_khong_hop_le", b, df, f"ngoài [{C.YEAR_MIN}, {C.YEAR_MAX}] hoặc thiếu")

    b = df
    df = _reject(df, df["brand"].isna() | df["model"].isna(), "brand_or_model_missing", rejected)
    step("thieu_hang_hoac_dong_xe", b, df)

    b = df
    df = _reject(df, df["seats"] > 16, "out_of_scope_bus", rejected)  # 45/54 chỗ: xe khách, không phải ô tô con
    step("xe_khach_>16_cho", b, df)

    b = df
    # dòng xe thuộc hãng khác rõ ràng (vd 'ecosport' bị gán 'mercedes-benz'): cặp hiếm (<=3 tin) mà dòng xe có >=90% tin ở hãng khác
    pair = df.groupby(["brand", "model"]).size().rename("n_pair").reset_index()
    tot = df.groupby("model").size().rename("n_model")
    top = pair.sort_values("n_pair", ascending=False).drop_duplicates("model").set_index("model")
    share = (top["n_pair"] / tot).rename("top_share")
    info = pair.join(share, on="model").join(top["brand"].rename("top_brand"), on="model").join(tot, on="model")
    mismatch = info[(info.n_pair <= 3) & (info.top_share >= 0.9) & (info.n_model >= 50) & (info.brand != info.top_brand)
                    & (info.model != "other")]
    key = df.set_index(["brand", "model"]).index.isin(mismatch.set_index(["brand", "model"]).index)
    df = _reject(df, pd.Series(key, index=df.index), "brand_model_mismatch", rejected)
    step("hang_dong_xe_khong_khop", b, df)

    # 3) Giá: chỉ loại khi xác minh được là sai, phần bị loại vì GIÁ nhưng đặc trưng còn dùng được -> tập chưa nhãn
    price_rej_start = len(rejected)
    b = df
    out_range = (df["price"] < C.PRICE_MIN_VND) | (df["price"] > C.PRICE_MAX_VND)
    df = _reject(df, out_range, "price_out_of_vnd_range", rejected)
    step("gia_ngoai_khoang_10tr_100ty", b, df, f"< {C.PRICE_MIN_VND:,} hoặc > {C.PRICE_MAX_VND:,} VND")

    b = df
    g = df.groupby(["brand", "model"])["price"]
    med, cnt = g.transform("median"), g.transform("size")
    ratio = df["price"] / med
    bad_ratio = (cnt >= C.PRICE_RATIO_MIN_GROUP) & ((ratio > C.PRICE_RATIO_LIMIT) | (ratio < 1 / C.PRICE_RATIO_LIMIT))
    rr = df.loc[bad_ratio, ["source", "source_id", "brand", "model", "year", "mileage_km", "price"]].copy()
    rr["reason"] = "price_ratio_vs_model_median"
    rr["model_median_price"] = med[bad_ratio]
    rejected.append(rr)
    df = df.loc[~bad_ratio]
    step("gia_lech_>15_lan_trung_vi_dong_xe", b, df, f"nhóm >= {C.PRICE_RATIO_MIN_GROUP} tin; so với trung vị cùng (hãng, dòng)")
    price_rejected = pd.concat(rejected[price_rej_start:]) if len(rejected) > price_rej_start else pd.DataFrame()

    # 4) Sửa mức trường (giữ dòng, đặt giá trị không hợp lệ thành thiếu)
    fixes = {}
    km_before = df["mileage_km"].copy()
    df["mileage_km"] = clean_mileage(df["mileage_km"])
    fixes["mileage_km_zero_or_missing_to_nan"] = int(((km_before == 0) & df["mileage_km"].isna()).sum())
    fixes["mileage_km_over_1M_to_nan"] = int((km_before > C.MAX_KM).sum())
    age = (C.REFERENCE_YEAR - df["year"]).clip(lower=1)
    too_fast = (df["mileage_km"] / age) > C.MAX_KM_PER_YEAR
    fixes["mileage_km_over_100k_per_year_to_nan"] = int(too_fast.sum())
    df.loc[too_fast, "mileage_km"] = np.nan
    bad_eng = (df["engine_capacity"] > 8) | ((df["engine_capacity"] < 0.6) & (df["fuel_type"] != "electric"))
    fixes["engine_capacity_implausible_to_nan"] = int(bad_eng.sum())
    df.loc[bad_eng, "engine_capacity"] = np.nan
    bad_seats = df["seats"] < 2
    fixes["seats_lt2_to_nan"] = int(bad_seats.sum())
    df.loc[bad_seats, "seats"] = np.nan
    rep["field_fixes"] = fixes

    # 5) group_id: tin giống nhau về mọi thuộc tính định danh xe (KHÔNG gồm km và giá) coi là cùng một xe đăng lại
    #    (người bán thường cập nhật km/giá khi đăng lại) -> cùng nhóm khi chia. Đây là định nghĩa BẢO THỦ:
    #    đôi khi gom cả những xe khác nhau nhưng cùng cấu hình, làm tập test khó hơn chứ không dễ hơn
    #    (xem reports/group_sensitivity.csv: khai báo nhóm theo cả km làm R² test cao hơn do rò rỉ tin đăng lại).
    sig_cols = ["source", "brand", "model", "variant", "year", "engine_capacity", "transmission",
                "fuel_type", "seats", "color", "province"]
    sig = df[sig_cols].astype(object).fillna("NA").astype(str).agg("|".join, axis=1)
    df["group_id"] = pd.factorize(sig)[0]

    df = df.reset_index(drop=True)
    labeled = df.copy()
    rejected_df = pd.concat(rejected, ignore_index=True) if rejected else pd.DataFrame(columns=["reason"])
    # Tập chưa nhãn: tin có đặc trưng hợp lệ nhưng giá bị loại vì không đáng tin -> bỏ nhãn giá, giữ đặc trưng
    unl = price_rejected.merge(raw.drop(columns=["price"]), on=["source", "source_id"], how="left", suffixes=("_x", ""))
    unl = unl[[c for c in data_sources.CANON_COLS if c != "price"]].copy() if len(unl) else pd.DataFrame(columns=[c for c in data_sources.CANON_COLS if c != "price"])
    unl["mileage_km"] = clean_mileage(unl["mileage_km"])
    unl = unl[unl["year"].between(C.YEAR_MIN, C.YEAR_MAX) & unl["brand"].notna() & unl["model"].notna()].reset_index(drop=True)
    unl[C.TARGET] = np.nan

    rep.update({"n_labeled": int(len(labeled)), "n_unlabeled": int(len(unl)), "n_rejected_total": int(len(rejected_df)),
                "rejected_by_reason": rejected_df["reason"].value_counts().to_dict(),
                "n_labeled_by_source": labeled.groupby("source").size().to_dict(),
                "n_groups": int(labeled["group_id"].nunique()), "max_group_size": int(labeled.groupby("group_id").size().max()),
                "price_vnd": {k: float(v) for k, v in labeled["price"].describe(percentiles=[.01, .5, .99]).items()},
                "missing_rate": labeled[C.INPUT_FEATURES].isna().mean().round(4).to_dict()})
    return labeled, unl, rejected_df, rep


def quality_checks(df: pd.DataFrame) -> list[str]:
    """Kiểm tra chất lượng dữ liệu đã làm sạch; trả danh sách vấn đề (rỗng = đạt)."""
    p = []
    if len(df) < C.MIN_ROWS:
        p.append(f"chỉ còn {len(df)} dòng (< {C.MIN_ROWS})")
    if df[C.TARGET].isna().any() or (df[C.TARGET] <= 0).any():
        p.append("còn giá thiếu/không dương")
    if not df[C.TARGET].between(C.PRICE_MIN_VND, C.PRICE_MAX_VND).all():
        p.append("còn giá ngoài khoảng hợp lệ")
    if not df["year"].between(C.YEAR_MIN, C.YEAR_MAX).all():
        p.append("còn năm sản xuất không hợp lệ")
    if (df["mileage_km"].dropna() > C.MAX_KM).any() or (df["mileage_km"].dropna() <= 0).any():
        p.append("còn số km không hợp lệ")
    if df.duplicated(subset=["source", "source_id"]).any():
        p.append("còn tin trùng source_id")
    if set(df.columns) & {"price_per_km", "log_price"}:
        p.append("có cột suy ra từ giá (nguy cơ rò rỉ nhãn)")
    return p


# ----------------------------------------------------------------------------- nạp dữ liệu
def load_clean(path=None, with_target=True, max_rows=None, refresh=False) -> pd.DataFrame:
    """Trả về bảng có nhãn đã làm sạch (cột: meta + INPUT_FEATURES + price).

    Dùng bản cache Data/processed/clean.csv nếu khớp phiên bản dữ liệu thô; ngược lại tạo lại từ Data/raw/.
    `max_rows`: lấy mẫu ngẫu nhiên theo NHÓM (seed cố định) để chạy thử/kiểm thử nhanh.
    """
    if path is not None:  # đường dẫn tường minh (CSV đã ở lược đồ chuẩn)
        df = pd.read_csv(path, low_memory=False)
    else:
        cache, info = C.PROCESSED / "clean.csv", C.PROCESSED / "data_info.json"
        df = None
        if cache.exists() and info.exists() and not refresh:
            try:
                if json.loads(info.read_text(encoding="utf-8")).get("data_version") == data_fingerprint():
                    df = pd.read_csv(cache, index_col="row_id", low_memory=False)
            except (json.JSONDecodeError, ValueError):
                df = None
        if df is None:
            df = build_dataset()[0]
            df.index.name = "row_id"
    if with_target and C.TARGET not in df.columns:
        raise ValueError(f"Thiếu cột mục tiêu '{C.TARGET}'")
    if with_target:
        df = df.dropna(subset=[C.TARGET])
        if len(df) < C.MIN_ROWS:
            raise ValueError(f"Chỉ còn {len(df)} dòng hợp lệ (< {C.MIN_ROWS}); kiểm tra dữ liệu.")
    if max_rows and len(df) > max_rows:
        rng = np.random.RandomState(C.RANDOM_STATE)
        gids = df["group_id"].unique()
        rng.shuffle(gids)
        sizes = df.groupby("group_id").size().reindex(gids).cumsum()
        df = df[df["group_id"].isin(sizes[sizes <= max_rows].index)]
    return df


def load_unlabeled() -> pd.DataFrame:
    """Tập chưa nhãn thật (tin có đặc trưng hợp lệ nhưng giá không đáng tin). Trống nếu chưa chạy preprocess."""
    p = C.PROCESSED / "unlabeled.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=C.INPUT_FEATURES)


# ----------------------------------------------------------------------------- chia dữ liệu theo nhóm
def split_indices(df, test_size=C.SERVE_TEST_SIZE, val_fraction=0.0, random_state=C.RANDOM_STATE):
    """Chia theo group_id: trả (train_idx, val_idx, test_idx) là nhãn index của df. val rỗng nếu val_fraction=0."""
    groups = df["group_id"].to_numpy()
    idx = np.arange(len(df))
    tr_val, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=random_state).split(idx, groups=groups))
    if val_fraction and val_fraction > 0:
        sub = groups[tr_val]
        tr, va = next(GroupShuffleSplit(1, test_size=val_fraction, random_state=random_state).split(tr_val, groups=sub))
        tr, va = tr_val[tr], tr_val[va]
    else:
        tr, va = tr_val, np.array([], dtype=int)
    return df.index[tr], df.index[va], df.index[te]


def get_split(test_size=C.SERVE_TEST_SIZE, random_state=C.RANDOM_STATE, df=None):
    """(X_train, X_test, y_train, y_test) theo nhóm, X chỉ gồm INPUT_FEATURES. Chưa fit gì cả."""
    df = load_clean() if df is None else df
    tr, _, te = split_indices(df, test_size, 0.0, random_state)
    X = df[C.INPUT_FEATURES]
    return X.loc[tr], X.loc[te], df.loc[tr, C.TARGET], df.loc[te, C.TARGET]


def get_split3(test_size=C.SERVE_TEST_SIZE, val_fraction=C.VAL_FRACTION, random_state=C.RANDOM_STATE, df=None):
    """Train / validation / test theo nhóm. Trả dict {'train','val','test'} -> (X, y, groups)."""
    df = load_clean() if df is None else df
    tr, va, te = split_indices(df, test_size, val_fraction, random_state)
    X = df[C.INPUT_FEATURES]
    return {k: (X.loc[i], df.loc[i, C.TARGET], df.loc[i, "group_id"]) for k, i in (("train", tr), ("val", va), ("test", te))}


# ----------------------------------------------------------------------------- CLI
def main():
    for d in (C.PROCESSED, C.SPLITS, C.MODELS, C.REPORTS):
        d.mkdir(parents=True, exist_ok=True)
    labeled, unl, rejected, rep = build_dataset()
    probs = quality_checks(labeled)
    if probs:
        raise SystemExit("Dữ liệu sau làm sạch chưa đạt: " + "; ".join(probs))
    labeled.index.name = "row_id"
    labeled.to_csv(C.PROCESSED / "clean.csv")
    unl.to_csv(C.PROCESSED / "unlabeled.csv", index=False)
    rejected.to_csv(C.PROCESSED / "rejected_rows.csv", index=False)

    rows = []
    for ts in C.TEST_SIZES:
        tr, va, te = split_indices(labeled, ts, C.VAL_FRACTION)
        gtr, gva, gte = (set(labeled.loc[i, "group_id"]) for i in (tr, va, te))
        assert not (gtr & gte) and not (gtr & gva) and not (gva & gte), "nhóm bị rò rỉ giữa các tập"
        rows.append({"test_size": ts, "train_pct": round((1 - ts) * (1 - C.VAL_FRACTION) * 100),
                     "val_pct": round((1 - ts) * C.VAL_FRACTION * 100), "test_pct": round(ts * 100),
                     "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                     "groups_shared_between_sets": 0})
        if abs(ts - C.SERVE_TEST_SIZE) < 1e-9:
            for nm, i in (("train", tr), ("val", va), ("test", te)):
                pd.Series(i, name="row_id").to_csv(C.SPLITS / f"{nm}_idx.csv", index=False)
    pd.DataFrame(rows).to_csv(C.SPLITS / "split_summary.csv", index=False)

    rep.update({"data_version": data_fingerprint(), "raw_files": {k: str(v.name) for k, v in C.RAW_FILES.items() if v.exists()},
                "input_features": C.INPUT_FEATURES, "target": C.TARGET, "price_unit": C.PRICE_UNIT,
                "random_state": C.RANDOM_STATE, "reference_year": C.REFERENCE_YEAR, "splits": rows,
                "thresholds": {"price_min_vnd": C.PRICE_MIN_VND, "price_max_vnd": C.PRICE_MAX_VND,
                               "price_ratio_limit": C.PRICE_RATIO_LIMIT, "price_ratio_min_group": C.PRICE_RATIO_MIN_GROUP,
                               "year_range": [C.YEAR_MIN, C.YEAR_MAX], "max_km": C.MAX_KM, "max_km_per_year": C.MAX_KM_PER_YEAR}})
    rep["n_rows_clean"] = rep["n_labeled"]
    (C.PROCESSED / "cleaning_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (C.PROCESSED / "data_info.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Thô: {rep['n_raw']:,} -> có nhãn sau làm sạch: {rep['n_labeled']:,} | chưa nhãn: {rep['n_unlabeled']:,} | bị loại: {rep['n_rejected_total']:,}")
    for s in rep["steps"]:
        print(f"  - {s['step']:<40s} loại {s['removed']:>6,}  còn {s['remaining']:>6,}")
    print("Sửa mức trường (không xóa dòng):", rep["field_fixes"])
    print("Lý do loại:", rep["rejected_by_reason"])
    print(f"Số nhóm tin: {rep['n_groups']:,} (nhóm lớn nhất: {rep['max_group_size']} tin)")
    print("Chia dữ liệu:", pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
