"""FastAPI: dự đoán giá xe + giám sát cơ bản.

Chạy (từ thư mục gốc project):  uvicorn app.main:app --reload
"""
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
app = FastAPI(title="Dự đoán giá xe", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
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


@app.get("/")
def index():
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
            "unit": p.info["price_unit"], "unit_note": p.info["price_unit_note"],
            "model_version": p.info["model_version"], "data_version": p.info["data_version"]}


@app.get("/api/metrics")
def metrics():
    """Chỉ số trên tập test của tỷ lệ chia đang được phục vụ (lấy từ lần chạy thực tế)."""
    p = get_predictor()
    return {"split": p.info["split"], "n_train": p.info["n_train"], "n_test": p.info["n_test"],
            "selection_criterion": p.info["selection_criterion"], "best_model": p.info["best_model"],
            "unit": p.info["price_unit"],
            "models": [{"model": k, "label": v["label"], **{m: round(v[m], 3) for m in ("cv_rmse", "MAE", "RMSE", "R2")}}
                       for k, v in p.info["models"].items()]}


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
