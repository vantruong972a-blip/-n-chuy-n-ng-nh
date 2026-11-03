"""FastAPI: dự đoán giá xe + giám sát cơ bản.

Chạy (từ thư mục gốc project):  uvicorn app.main:app --reload
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
from inference import InvalidInput, ModelNotAvailable, Predictor  # noqa: E402
from monitoring import Monitor, Timer  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
# Giao diện React (frontend/): nếu đã build thì phục vụ ở "/", giao diện HTML cũ vẫn truy cập được ở "/cu".
FRONTEND_DIST = Path(os.environ.get("MLOPS_FRONTEND_DIST", ROOT / "frontend" / "dist"))
app = FastAPI(title="Dự đoán giá xe", version="1.1")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")
monitor = Monitor()
_state = {"predictor": None}


def get_predictor() -> Predictor:
    """Nạp lười: thiếu mô hình thì trả 503 rõ ràng thay vì làm sập ứng dụng."""
    if _state["predictor"] is None:
        try:
            _state["predictor"] = Predictor()
        except ModelNotAvailable as e:
            raise HTTPException(503, str(e))
    return _state["predictor"]


class PredictIn(BaseModel):
    values: Dict[str, Any]
    model: Optional[str] = None


class FeedbackIn(BaseModel):
    prediction_id: str
    actual_price: float


@app.exception_handler(HTTPException)
async def _http_exc(request: Request, exc: HTTPException):
    if request.url.path.startswith("/api") and exc.status_code >= 400:
        monitor.record_error(request.url.path, exc.status_code, exc.detail)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


def _spa_index():
    f = FRONTEND_DIST / "index.html"
    return FileResponse(f if f.exists() else STATIC / "index.html")


@app.get("/")
def index():
    return _spa_index()


@app.get("/cu")
def index_legacy():
    """Giao diện HTML cũ (giữ lại để đối chiếu/dự phòng)."""
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    try:
        p = get_predictor()
        return {"status": "ok", "models": list(p.models), "model_version": p.info["model_version"]}
    except HTTPException as e:
        return JSONResponse({"status": "model_unavailable", "detail": e.detail}, status_code=503)


@app.get("/api/meta")
def meta():
    p = get_predictor()
    return {"features": p.feature_meta(), "models": p.labels, "default_model": p.default_model,
            "unit": p.info["price_unit"], "unit_note": p.info["price_unit_note"], "required": p.required,
            "model_version": p.info["model_version"], "data_version": p.info["data_version"],
            "brand_models": p.ref.get("brand_models", {})}


@app.get("/api/metrics")
def metrics():
    """Chỉ số trên tập test của tỷ lệ chia đang được phục vụ (lấy từ lần chạy thực tế)."""
    p = get_predictor()
    return {"split": p.info["split"], "n_train": p.info["n_train"], "n_test": p.info["n_test"],
            "selection_criterion": p.info["selection_criterion"], "best_model": p.info["best_model"],
            "unit": p.info["price_unit"], "unit_note": p.info["price_unit_note"], "n_val": p.info.get("n_val"),
            "split_strategy": p.info.get("split_strategy"),
            "models": [{"model": k, "label": v["label"], "config_used": v.get("config_used"),
                        **{m: (None if v.get(m) is None else round(v[m], 3)) for m in ("cv_rmse", "val_RMSE", "MAE", "RMSE", "R2", "MAPE", "medAPE")}}
                       for k, v in p.info["models"].items()]}


@app.get("/api/model-info")
def model_info():
    """Thông tin phiên bản mô hình đang phục vụ (đọc từ models/model_info.json do train_supervised.py ghi)."""
    p = get_predictor()
    i = p.info
    keys = ("model_version", "trained_at", "data_version", "n_rows_clean", "n_raw", "split", "split_strategy", "n_train", "n_val",
            "n_test", "target", "target_transform", "price_unit", "price_unit_note", "reference_year", "cv", "tuned",
            "tuning_iterations", "selection_criterion", "best_model", "libraries")
    return {**{k: i.get(k) for k in keys}, "serving_models": list(p.models),
            "models": {k: {"label": v["label"], "config_used": v.get("config_used"), "best_params": v.get("best_params", {})}
                       for k, v in i["models"].items()}}


@app.get("/api/metrics/all")
def metrics_all():
    f = C.REPORTS / "supervised_results.csv"
    if not f.exists():
        raise HTTPException(404, "Chưa có reports/supervised_results.csv. Hãy chạy train_supervised.py --all")
    return pd.read_csv(f).round(3).to_dict(orient="records")


@app.post("/api/predict")
def predict(body: PredictIn):
    p = get_predictor()
    with Timer() as t:
        try:
            r = p.predict(body.values, body.model)
        except InvalidInput as e:
            monitor.record_error("/api/predict", 422, e)
            raise HTTPException(422, e.errors)
    monitor.record_prediction(r["prediction_id"], r["model"], r["model_version"], r.pop("inputs"), r["price"], t.ms)
    return r


@app.post("/api/feedback")
def feedback(body: FeedbackIn):
    """Gửi giá thực tế đến sau để theo dõi sai số theo thời gian."""
    if not monitor.has_prediction(body.prediction_id):
        raise HTTPException(404, "Không tìm thấy prediction_id")
    monitor.record_feedback(body.prediction_id, body.actual_price)
    return {"status": "recorded"}


@app.get("/api/monitoring")
def monitoring():
    p = get_predictor()
    return monitor.summary(p.ref, p.info["model_version"])


@app.get("/{path:path}", include_in_schema=False)
def spa_fallback(path: str):
    """Các đường dẫn của React Router (vd /dinh-gia): trả index.html. Đường dẫn /api/* không tồn tại vẫn trả 404 dạng JSON."""
    if path.startswith(("api/", "assets/", "static/")) or not (FRONTEND_DIST / "index.html").exists():
        raise HTTPException(404, "Không tìm thấy")
    f = (FRONTEND_DIST / path).resolve()
    if path and f.is_file() and FRONTEND_DIST.resolve() in f.parents:   # file tĩnh ở gốc dist (favicon...)
        return FileResponse(f)
    return FileResponse(FRONTEND_DIST / "index.html")
