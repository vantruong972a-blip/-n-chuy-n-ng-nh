import numpy as np
import pandas as pd
import pytest

import config as C
from features import FeatureEngineer, build_preprocessor
from preprocess import get_split, get_split3, split_indices


@pytest.mark.parametrize("ts", [0.4, 0.3, 0.2, 0.1])
def test_group_split_sizes_and_no_group_leak(cleaned, ts):
    s = get_split3(ts, df=cleaned)
    n = sum(len(v[0]) for v in s.values())
    assert n == len(cleaned)
    assert abs(len(s["test"][0]) / n - ts) < 0.03
    g = {k: set(v[2]) for k, v in s.items()}
    assert not g["train"] & g["test"] and not g["train"] & g["val"] and not g["val"] & g["test"]
    for k in s:
        assert C.TARGET not in s[k][0].columns and len(s[k][0]) == len(s[k][1])
    assert set(s["train"][0].columns) == set(C.INPUT_FEATURES)


def test_split_reproducible_and_seed_sensitive(cleaned):
    a = get_split(0.2, df=cleaned)[0].index
    assert list(a) == list(get_split(0.2, df=cleaned)[0].index)
    assert list(a) != list(get_split(0.2, random_state=7, df=cleaned)[0].index)


def test_repeated_listings_stay_together(cleaned):
    tr, va, te = split_indices(cleaned, 0.2, C.VAL_FRACTION)
    multi = cleaned.groupby("group_id").filter(lambda g: len(g) > 1)
    assert len(multi) > 100     # dữ liệu thật có tin đăng lặp lại cùng một xe
    where = {i: "tr" for i in tr} | {i: "va" for i in va} | {i: "te" for i in te}
    per_group = multi.assign(s=multi.index.map(where)).groupby("group_id")["s"].nunique()
    assert (per_group == 1).all()


def test_feature_engineering_values_and_no_price_use():
    X = pd.DataFrame([{"brand": "toyota", "model": "vios", "year": 2018, "mileage_km": 60000},
                      {"brand": "kia", "model": "morning", "year": 2022, "mileage_km": 0}])
    F = FeatureEngineer().fit_transform(X)
    assert F.loc[0, "car_age"] == C.REFERENCE_YEAR - 2018 and F.loc[0, "km_per_year"] == 60000 / 4
    assert F.loc[0, "brand_model"] == "toyota|vios" and F.loc[0, "condition"] == "used"
    assert F.loc[1, "km_reported"] == 0 and np.isnan(F.loc[1, "km_per_year"]) and F.loc[1, "condition"] == "km_unknown"
    assert "price" not in F.columns


def test_preprocessor_fit_on_train_only(cleaned):
    X_tr, X_te, _, _ = get_split(0.2, df=cleaned)
    pre = build_preprocessor().fit(X_tr)
    A_tr, A_te = pre.transform(X_tr), pre.transform(X_te)
    n_num = len(pre.named_steps["ct"].transformers_[0][2])
    assert np.allclose(A_tr[:, :n_num].mean(0), 0, atol=1e-6)          # chuẩn hóa theo train
    assert not np.allclose(A_te[:, :n_num].mean(0), 0, atol=1e-6)      # test không bị fit lại
    assert A_tr.shape[1] == A_te.shape[1] and np.isfinite(A_te).all()


def test_preprocessor_handles_unseen_and_missing(cleaned):
    X_tr, _, _, _ = get_split(0.2, df=cleaned)
    pre = build_preprocessor().fit(X_tr)
    odd = pd.DataFrame([{"brand": "hãng-chưa-từng-thấy", "model": "zzz", "year": 2015}])
    assert np.isfinite(pre.transform(odd)).all()


def test_missing_engine_filled_by_model_median_not_global(cleaned):
    """Bỏ trống dung tích: phải điền theo dòng xe (Vios ~1,5 lít), không phải trung vị toàn bộ dữ liệu (~2,0 lít)."""
    X_tr, _, _, _ = get_split(0.2, df=cleaned)
    pre = build_preprocessor().fit(X_tr)
    F = pre.named_steps["fill"].transform(pre.named_steps["fe"].transform(pd.DataFrame([{"brand": "toyota", "model": "vios", "year": 2018}])))
    assert F.loc[0, "engine_capacity"] == pytest.approx(1.5, abs=0.1) and F.loc[0, "seats"] in (4, 5)
    rich = pre.named_steps["fill"].transform(pre.named_steps["fe"].transform(pd.DataFrame([{"brand": "toyota", "model": "xe-lạ-chưa-có", "year": 2018}])))
    assert np.isfinite(rich.loc[0, "engine_capacity"])        # dòng xe lạ -> trung vị theo hãng
