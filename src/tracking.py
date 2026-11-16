"""Bọc MLflow. Nếu chưa cài mlflow thì các hàm thành no-op (có cảnh báo), pipeline vẫn chạy."""
import contextlib
import warnings
from pathlib import Path

import config as C

try:
    import mlflow
    import mlflow.sklearn
    AVAILABLE = True
except ImportError:  # pragma: no cover
    mlflow = None
    AVAILABLE = False


def setup(enabled=True):
    if not enabled:
        return False
    if not AVAILABLE:
        warnings.warn("Chưa cài mlflow -> không ghi MLflow. Chạy: pip install mlflow")
        return False
    C.MLRUNS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + C.MLFLOW_DB.as_posix())
    if mlflow.get_experiment_by_name(C.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(C.MLFLOW_EXPERIMENT, artifact_location=C.MLRUNS.as_uri())
    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
    return True


@contextlib.contextmanager
def run(enabled, run_name, tags=None):
    if not (enabled and AVAILABLE):
        yield None
        return
    with mlflow.start_run(run_name=run_name) as r:
        if tags:
            mlflow.set_tags({k: str(v) for k, v in tags.items()})
        yield r


def _clean_params(params):
    return {str(k): ("None" if v is None else v) for k, v in params.items()}


def log(enabled, params=None, metrics=None, model=None, artifacts=None, model_name=None):
    """Ghi vào run đang mở: tham số, chỉ số, file kết quả (artifacts) và mô hình. No-op khi tắt MLflow."""
    if not (enabled and AVAILABLE):
        return
    if params:
        mlflow.log_params(_clean_params(params))
    if metrics:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
    for a in artifacts or []:
        if Path(a).exists():
            mlflow.log_artifact(str(a))
    if model is not None:
        try:
            mlflow.sklearn.log_model(model, name=model_name or "model", serialization_format="cloudpickle")
        except TypeError:  # mlflow 2.x dùng artifact_path
            mlflow.sklearn.log_model(model, artifact_path=model_name or "model", serialization_format="cloudpickle")


REGISTRY_MODEL = "car-price-model"
CHAMPION = "champion"


def register_champion(model_version: str, algorithm: str, enabled=True):
    """Đăng ký mô hình đang được API phục vụ vào MLflow Model Registry và gán bí danh 'champion'.

    Gọi sau khi mô hình mới đã qua kiểm tra và được nâng cấp. Không bao giờ làm hỏng huấn luyện: lỗi chỉ cảnh báo.
    Trả {name, version, alias} hoặc None.
    """
    if not (enabled and AVAILABLE):
        return None
    try:
        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name(C.MLFLOW_EXPERIMENT)
        runs = client.search_runs([exp.experiment_id], max_results=5, order_by=["attributes.start_time DESC"],
                                  filter_string=f"tags.model_version = '{model_version}' and params.algorithm = '{algorithm}'")
        if not runs:
            return None
        run_id = runs[0].info.run_id
        uri = f"runs:/{run_id}/{algorithm}"
        if hasattr(mlflow, "search_logged_models"):   # MLflow >= 3: mô hình là LoggedModel riêng
            lm = [m for m in mlflow.search_logged_models(experiment_ids=[exp.experiment_id], filter_string=f"name = '{algorithm}'",
                                                         output_format="list") if m.source_run_id == run_id]
            if lm:
                uri = f"models:/{lm[0].model_id}"
        mv = mlflow.register_model(uri, REGISTRY_MODEL, tags={"model_version": model_version, "algorithm": algorithm})
        client.set_registered_model_alias(REGISTRY_MODEL, CHAMPION, mv.version)
        return {"name": REGISTRY_MODEL, "version": str(mv.version), "alias": CHAMPION}
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"Không đăng ký được vào MLflow Model Registry: {e}")
        return None
