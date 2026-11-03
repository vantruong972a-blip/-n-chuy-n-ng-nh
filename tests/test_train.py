import json

import joblib
import numpy as np
import pandas as pd
import pytest

import config as C
import train_supervised as T


def test_results_file(trained):
    df = pd.read_csv(C.REPORTS / "supervised_results.csv")
    need = {"split", "n_train", "n_val", "n_test", "model", "config_used", "cv_rmse", "val_RMSE_baseline", "val_RMSE_tuned",
            "val_RMSE", "MAE", "RMSE", "R2", "MAPE", "medAPE", "train_seconds", "selected_best"}
    assert need <= set(df.columns)
    assert set(df.model) == {"linear_regression", "random_forest", "xgboost"}
    assert np.isfinite(df[["MAE", "RMSE", "R2"]].values).all()
    assert df.n_test.nunique() == 1 and df.n_train.nunique() == 1       # cả 3 mô hình dùng cùng một test/train
    assert df.selected_best.sum() == 1
    assert df.loc[df.selected_best, "val_RMSE"].iloc[0] == df.val_RMSE.min()   # chọn theo validation, không theo test
    # cấu hình dùng là cấu hình có RMSE validation thấp hơn
    for _, r in df.iterrows():
        if r.config_used == "tuned":
            assert r.val_RMSE_tuned < r.val_RMSE_baseline


def test_evaluation_reports_written(trained):
    for f in ("model_comparison_table.csv", "model_comparison_table.md", "feature_importance.csv", "performance_by_group.csv",
              "worst_errors.csv", "target_transform_ablation.csv", "reference_baseline.csv", "model_comparison.png"):
        assert (C.REPORTS / f).exists(), f
    imp = pd.read_csv(C.REPORTS / "feature_importance.csv")
    cols = imp[imp.kind == "column"]
    assert set(cols.feature) == set(C.INPUT_FEATURES) and (cols.groupby("model").size() == len(C.INPUT_FEATURES)).all()
    assert (imp.kind == "group").any()
    assert {"brand_top10", "age_band", "price_band"} <= set(pd.read_csv(C.REPORTS / "performance_by_group.csv").group_by)


def test_model_info_and_save_load(trained, cleaned):
    info = json.loads((C.MODELS / "model_info.json").read_text(encoding="utf-8"))
    for k in ("model_version", "data_version", "features", "libraries", "split", "price_unit", "reference_year", "n_val"):
        assert k in info
    assert info["price_unit"] == "VND" and info["features"] == C.INPUT_FEATURES
    X = cleaned[C.INPUT_FEATURES].head(10)
    for name in info["models"]:
        p1 = joblib.load(C.MODELS / f"{name}.joblib").predict(X)
        p2 = joblib.load(C.MODELS / "versions" / info["model_version"] / f"{name}.joblib").predict(X)
        assert np.isfinite(p1).all() and (p1 > 0).all() and np.allclose(p1, p2)


def test_serving_pipeline_accepts_raw_rows_with_missing_values(trained):
    info = json.loads((C.MODELS / "model_info.json").read_text(encoding="utf-8"))
    pipe = joblib.load(C.MODELS / f"{info['best_model']}.joblib")
    p = pipe.predict(pd.DataFrame([{"brand": "toyota", "model": "vios", "year": 2018}]))
    assert np.isfinite(p).all() and p[0] > 0


def test_candidate_gate_blocks_bad_model(trained, cleaned):
    info = json.loads((C.MODELS / "model_info.json").read_text(encoding="utf-8"))
    ctx = {"X_te": cleaned[C.INPUT_FEATURES].head(50), "y_te": cleaned[C.TARGET].head(50), "y_tr": cleaned[C.TARGET]}
    ok, _ = T._check_candidate(info, ctx)
    assert ok
    bad = json.loads(json.dumps(info))
    bad["models"][bad["best_model"]]["R2"] = -1.0
    ok, why = T._check_candidate(bad, ctx)
    assert not ok and "tham chiếu" in why


def test_old_models_archived_before_overwrite(cleaned, monkeypatch, tmp_path):
    """Lần huấn luyện thứ hai: bản đang phục vụ phải được lưu vào models/archive trước khi bị thay (thư mục riêng, không đụng test khác)."""
    monkeypatch.setattr(C, "MODELS", tmp_path / "models")
    monkeypatch.setattr(C, "REPORTS", tmp_path / "reports")
    kw = dict(model_names=["linear_regression"], fast=True, use_mlflow=False, max_rows=1500, n_iter=1, ablate_target=False)
    T.run([0.2], **kw)
    old = json.loads((C.MODELS / "model_info.json").read_text(encoding="utf-8"))["model_version"]
    import time
    time.sleep(1.1)   # model_version có độ phân giải giây
    T.run([0.2], **kw)
    new = json.loads((C.MODELS / "model_info.json").read_text(encoding="utf-8"))["model_version"]
    assert new != old and (C.MODELS / "archive" / old / "model_info.json").exists()


def test_max_rows_subsample_never_splits_a_group(cleaned):
    from preprocess import load_clean
    sub = load_clean(max_rows=2000)
    assert 0 < len(sub) <= 2000
    full_sizes = cleaned.groupby("group_id").size()
    assert (sub.groupby("group_id").size() == full_sizes.reindex(sub.group_id.unique()).values).all()
