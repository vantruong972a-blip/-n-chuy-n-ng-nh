import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import config as C
import data_sources
from preprocess import build_dataset, quality_checks


def test_raw_files_unchanged_checksum():
    sums = dict(reversed(l.split()) for l in (C.RAW_DIR / "CHECKSUMS.sha256").read_text().splitlines() if l.strip())
    for name, want in sums.items():
        assert hashlib.sha256((C.RAW_DIR / name).read_bytes()).hexdigest() == want, f"{name} đã bị sửa"


def test_enough_real_records(cleaned):
    assert len(cleaned) >= 10_000
    assert set(cleaned["source"]) == {"bonbanh", "chotot"}
    assert quality_checks(cleaned) == []


def test_schema_and_ranges(cleaned):
    assert set(C.INPUT_FEATURES) | {C.TARGET, "group_id", "source", "source_id"} <= set(cleaned.columns)
    assert cleaned[C.TARGET].between(C.PRICE_MIN_VND, C.PRICE_MAX_VND).all()
    assert cleaned["year"].between(C.YEAR_MIN, C.YEAR_MAX).all()
    assert (cleaned["mileage_km"].dropna() > 0).all() and (cleaned["mileage_km"].dropna() <= C.MAX_KM).all()
    assert not cleaned.duplicated(subset=["source", "source_id"]).any()
    assert cleaned["brand"].notna().all() and cleaned["model"].notna().all()


def test_no_target_leakage_columns(cleaned):
    assert not [c for c in C.INPUT_FEATURES if "price" in c]
    assert C.TARGET not in C.INPUT_FEATURES


def test_cleaning_report_is_consistent(cleaned):
    rep = json.loads((C.PROCESSED / "cleaning_report.json").read_text(encoding="utf-8"))
    assert rep["n_raw"] - sum(s["removed"] for s in rep["steps"]) == rep["n_labeled"] == len(cleaned)
    rej = pd.read_csv(C.PROCESSED / "rejected_rows.csv")
    assert len(rej) == rep["n_rejected_total"] and rej["reason"].notna().all()
    out = rej[rej.reason == "price_out_of_vnd_range"]["price"]      # mỗi dòng bị loại vì giá phải thật sự ngoài khoảng
    assert ((out < C.PRICE_MIN_VND) | (out > C.PRICE_MAX_VND)).all()


def test_unlabeled_has_no_price_and_is_disjoint(cleaned):
    unl = pd.read_csv(C.PROCESSED / "unlabeled.csv")
    assert unl[C.TARGET].isna().all()
    key = lambda d: set(zip(d.source, d.source_id.astype(str)))  # noqa: E731
    assert not key(unl) & key(cleaned.assign(source_id=cleaned.source_id.astype(str)))


def _row(**kw):
    base = dict(source="chotot", source_id="1", brand="toyota", model="vios", variant=np.nan, body_type="sedan", year=2018,
                mileage_km=50000.0, engine_capacity=1.5, transmission="automatic", fuel_type="gasoline", drivetrain=np.nan,
                seats=5.0, origin="domestic", province="ha-noi", color=np.nan, price=450e6)
    base.update(kw)
    return base


def test_cleaning_rules_on_small_hand_made_frame():
    """Kiểm thử LOGIC làm sạch trên bảng nhỏ dựng tay (không phải dữ liệu huấn luyện)."""
    rows = [_row(source_id=str(i), price=400e6 + i * 5e6, mileage_km=40000.0 + i) for i in range(12)]
    rows += [_row(source_id="d1", price=420e6, mileage_km=40001.0),                      # trùng mọi thuộc tính với tin i=... (xem dưới)
             _row(source_id="p0", price=0), _row(source_id="p1", price=5),                  # giá không hợp lệ
             _row(source_id="p2", price=12_345_678_000),                                    # lệch >15 lần trung vị dòng xe
             _row(source_id="y1", year=2035), _row(source_id="y2", year=1900),             # năm vô lý
             _row(source_id="k1", mileage_km=4294967295.0),                                 # km tràn số -> NaN, GIỮ dòng
             _row(source_id="k2", mileage_km=0.0),                                          # 0 km = chưa khai báo -> NaN, GIỮ dòng
             _row(source_id="rich", brand="rolls-royce", model="phantom", price=68e9)]      # xe sang giá cao nhưng hợp lệ -> GIỮ
    labeled, unl, rej, rep = build_dataset(pd.DataFrame(rows))
    kept = set(labeled.source_id)
    assert {"p0", "p1", "p2", "y1", "y2"}.isdisjoint(kept)
    assert {"k1", "k2", "rich"} <= kept
    assert labeled.loc[labeled.source_id.isin(["k1", "k2"]), "mileage_km"].isna().all()
    assert set(rej.reason) >= {"price_missing_or_nonpositive", "price_out_of_vnd_range", "price_ratio_vs_model_median", "year_invalid"}
    assert "p2" in set(unl.source_id) and C.TARGET in unl and unl[C.TARGET].isna().all()   # giá lỗi -> tin chưa nhãn


def test_province_parsing_and_normalisation():
    assert data_sources.province_from_url("https://xe.chotot.com/mua-ban-oto-cu-sdca1-quan-8-tp-ho-chi-minh/9.htm#px=1") == "tp-ho-chi-minh"
    assert data_sources.province_from_url("https://x/mua-ban-oto-cu-sdca1-thi-xa-di-an-binh-duong/9.htm") == "binh-duong"
    assert data_sources.norm_model("C Class") == "c" and data_sources.norm_brand("Mercedes Benz") == "mercedes-benz"
    assert data_sources.norm_brand("Mercedes") == "mercedes-benz"
