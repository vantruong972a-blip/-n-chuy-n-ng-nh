import json

import joblib
import numpy as np
import pandas as pd

import config as C
import roles as R
from features import DepreciationFeatures, FeatureEngineer, GroupMedianImputer
from model_registry import ROLES, make_pipeline


def test_linear_pipeline_uses_depreciation_features_and_others_do_not():
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    lin = make_pipeline(LinearRegression(), name="linear_regression")
    rf = make_pipeline(RandomForestRegressor(n_estimators=3), name="random_forest")
    assert [t[0] for t in lin.named_steps["pre"].named_steps["ct"].transformers] == ["dep", "num", "cat"]
    assert [t[0] for t in rf.named_steps["pre"].named_steps["ct"].transformers] == ["num", "cat"]


def test_depreciation_features_are_finite_and_have_per_brand_slopes(cleaned):
    X = cleaned[C.INPUT_FEATURES].head(3000)
    Z = FeatureEngineer().fit_transform(X)
    Z = GroupMedianImputer().fit(Z).transform(Z)
    d = DepreciationFeatures(min_brand=50).fit(Z)
    out = d.transform(Z)
    assert np.isfinite(out.to_numpy()).all()
    assert any(c.startswith("age@") for c in out.columns) and any(c.startswith("lkm@") for c in out.columns)
    assert list(d.get_feature_names_out()) == list(out.columns)


def test_roles_files_and_info(trained):
    info = json.loads((C.MODELS / "model_info.json").read_text(encoding="utf-8"))
    assert set(info["roles"]) == {"linear_regression", "random_forest", "xgboost"}
    for m, r in info["roles"].items():
        assert r["title"] == ROLES[m]["title"] and r["n_segments"] > 0 and np.isfinite(r["val_RMSLE"])
        assert 0 <= r["wins_rmsle"] <= r["n_segments"]
    # mỗi phân khúc có đúng một mô hình thắng => tổng số thắng = số phân khúc
    assert sum(r["wins_rmsle"] for r in info["roles"].values()) == next(iter(info["roles"].values()))["n_segments"]
    rec = info["recommend_by_band"]
    assert rec and all(b["model"] in info["roles"] for b in rec)
    tab = pd.read_csv(C.REPORTS / "model_roles.csv")
    assert {"model", "segment_type", "segment", "n", "RMSLE", "medAPE", "bias_pct"} <= set(tab.columns)
    assert (C.REPORTS / "model_roles.md").exists() and (C.REPORTS / "depreciation_by_brand.csv").exists()
    dep = pd.read_csv(C.REPORTS / "depreciation_by_brand.csv")
    assert (dep.per_year_pct < 0).all() and (dep.per_10k_km_pct <= 0.5).all()      # xe già hơn / chạy nhiều hơn thì rẻ hơn


def test_recommend_keeps_default_unless_clearly_better():
    y = pd.Series(np.full(400, 5e8))
    X = pd.DataFrame({"year": [2018] * 400, "mileage_km": [50000.0] * 400})
    rng = np.random.default_rng(0)
    a = y * np.exp(rng.normal(0, 0.20, 400))
    b = y * np.exp(rng.normal(0, 0.195, 400))        # chỉ tốt hơn ~2,5%: dưới ngưỡng 3%
    res = R.build_roles(X, y, {"xgboost": a.to_numpy(), "random_forest": b.to_numpy()}, "xgboost")
    assert all(r["model"] == "xgboost" for r in res["recommend_by_band"])
    c = y * np.exp(rng.normal(0, 0.10, 400))         # tốt hơn rõ rệt
    res = R.build_roles(X, y, {"xgboost": a.to_numpy(), "random_forest": c.to_numpy()}, "xgboost")
    assert any(r["model"] == "random_forest" for r in res["recommend_by_band"])
