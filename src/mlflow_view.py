"""Đọc MLflow (chỉ đọc) để hiển thị trong web. Không ghi, không chạy huấn luyện.

Trả {"available": False, "reason": ...} khi chưa cài mlflow / chưa có mlflow.db / chưa có thí nghiệm,
để giao diện báo đúng lý do thay vì lỗi 500.
"""
import datetime as dt

import config as C

METRIC_KEYS = ("val_RMSE", "test_RMSE", "test_MAE", "test_R2", "test_MAPE", "test_medAPE", "cv_rmse", "train_seconds")
PARAM_KEYS = ("algorithm", "split_strategy", "tuning", "target_transform", "train_val_test", "cv", "data_version", "random_state")
MAX_RUNS = 60


def _iso(ms):
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).isoformat(timespec="seconds") if ms else None


def _off(reason):
    return {"available": False, "reason": reason, "experiment": None, "runs": [], "registry": None}


def mlflow_summary():
    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError:
        return _off("Chưa cài thư viện mlflow trên máy chủ (pip install mlflow).")
    if not C.MLFLOW_DB.exists():
        return _off("Chưa có file mlflow.db. Hãy huấn luyện: python src\\train_supervised.py")
    client = MlflowClient(tracking_uri="sqlite:///" + C.MLFLOW_DB.as_posix(), registry_uri="sqlite:///" + C.MLFLOW_DB.as_posix())
    exp = client.get_experiment_by_name(C.MLFLOW_EXPERIMENT)
    if exp is None:
        return _off(f"mlflow.db chưa có thí nghiệm '{C.MLFLOW_EXPERIMENT}'. Hãy huấn luyện lại.")
    runs = client.search_runs([exp.experiment_id], max_results=MAX_RUNS, order_by=["attributes.start_time DESC"])
    out = []
    for r in runs:
        d = r.data
        out.append({"run_id": r.info.run_id, "name": r.info.run_name, "status": r.info.status, "started_at": _iso(r.info.start_time),
                    "model_version": d.tags.get("model_version"), "split": d.tags.get("split"),
                    "selected_best": d.tags.get("selected_best") == "True", "served": d.tags.get("served") == "True",
                    "metrics": {k: d.metrics.get(k) for k in METRIC_KEYS if k in d.metrics},
                    "params": {k: d.params.get(k) for k in PARAM_KEYS if k in d.params},
                    "hyperparameters": {k[3:]: v for k, v in d.params.items() if k.startswith("hp_")}})
    registry = None
    try:
        rm = client.get_registered_model("car-price-model")
        versions = sorted(client.search_model_versions("name='car-price-model'"), key=lambda v: -int(v.version))
        al = {a: str(v) for a, v in (rm.aliases or {}).items()}
        registry = {"name": rm.name, "aliases": al,
                    "versions": [{"version": str(v.version), "created_at": _iso(v.creation_timestamp), "run_id": v.run_id,
                                  "aliases": sorted(a for a, ver in al.items() if ver == str(v.version)), "model_version": (v.tags or {}).get("model_version"),
                                  "algorithm": (v.tags or {}).get("algorithm")} for v in versions[:20]]}
    except mlflow.exceptions.MlflowException:
        registry = None   # chưa có mô hình nào được đăng ký
    return {"available": True, "reason": None, "experiment": {"name": exp.name, "id": exp.experiment_id}, "runs": out,
            "registry": registry}
