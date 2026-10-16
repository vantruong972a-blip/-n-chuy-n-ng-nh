import json

import joblib
import numpy as np
import pandas as pd

import config as C


def test_results_file(trained):
    f = C.REPORTS / "supervised_results.csv"
    assert f.exists()
    df = pd.read_csv(f)
    need = {"split", "n_train", "n_test", "model", "cv_rmse", "MAE", "RMSE", "R2", "train_seconds", "selected_best"}
    assert need <= set(df.columns)
    assert set(df.split) == {"70/30", "80/20"}
    assert np.isfinite(df[["MAE", "RMSE", "R2"]].values).all()
    assert (df.groupby("split").selected_best.sum() == 1).all()
    # mô hình được chọn phải là mô hình có CV-RMSE thấp nhất (không chọn theo test)
    for _, g in df.groupby("split"):
        assert g.loc[g.selected_best, "cv_rmse"].iloc[0] == g.cv_rmse.min()


def test_model_info_and_artifacts(trained):
    info = json.loads((C.MODELS / "model_info.json").read_text(encoding="utf-8"))
    for k in ("model_version", "data_version", "features", "libraries", "split", "price_unit"):
        assert k in info
    assert (C.MODELS / "reference_stats.json").exists()
    name = info["best_model"]
    pipe = joblib.load(C.MODELS / f"{name}.joblib")
    X = pd.DataFrame([{c: (1.0 if c in info["numeric_features"] else None) for c in info["features"]}])
    X.loc[0, info["categorical_features"]] = [json.loads((C.MODELS / 'reference_stats.json').read_text(encoding='utf-8'))["categorical"][c]["default"] for c in info["categorical_features"]]
    assert np.isfinite(pipe.predict(X)).all()  # Pipeline nhận dữ liệu thô, tự tiền xử lý
