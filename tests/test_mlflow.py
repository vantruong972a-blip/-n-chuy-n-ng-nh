import pytest

import config as C
import train_supervised as T

mlflow = pytest.importorskip("mlflow")


def test_mlflow_logs_params_metrics_and_model(cleaned, monkeypatch, tmp_path):
    for k, v in {"MODELS": tmp_path / "m", "REPORTS": tmp_path / "r", "MLRUNS": tmp_path / "mlruns", "MLFLOW_DB": tmp_path / "mlflow.db"}.items():
        monkeypatch.setattr(C, k, v)
    T.run([0.2], model_names=["linear_regression"], fast=True, use_mlflow=True, max_rows=1500, n_iter=1, ablate_target=False)
    mlflow.set_tracking_uri("sqlite:///" + C.MLFLOW_DB.as_posix())
    exp = mlflow.get_experiment_by_name(C.MLFLOW_EXPERIMENT)
    assert exp is not None
    runs = mlflow.search_runs([exp.experiment_id])
    assert len(runs) >= 1
    r = runs.iloc[0]
    for col in ("params.algorithm", "params.data_version", "params.random_state", "params.features", "params.split_strategy",
                "metrics.test_RMSE", "metrics.test_R2", "metrics.val_RMSE", "tags.model_version"):
        assert col in runs.columns and r[col] == r[col], col          # tồn tại và không NaN
    client = mlflow.MlflowClient()
    names = {f.path for f in client.list_artifacts(r.run_id)}
    assert "data_info.json" in names and "cleaning_report.json" in names
    if hasattr(client, "search_logged_models"):                    # MLflow >= 3
        assert client.search_logged_models([exp.experiment_id])    # mô hình đã được log
