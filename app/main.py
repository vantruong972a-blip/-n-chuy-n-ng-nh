"""FastAPI: dự đoán giá xe + giám sát cơ bản.

Chạy (từ thư mục gốc project):  uvicorn app.main:app --reload
"""
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Literal

from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
from inference import InvalidInput, ModelNotAvailable, Predictor  # noqa: E402
from monitoring import Monitor, Timer  # noqa: E402
from recommend import DataNotAvailable, load_recommender  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
# Giao diện React (frontend/): nếu đã build thì phục vụ ở "/", giao diện HTML cũ vẫn truy cập được ở "/cu".
IMAGES_DIR = C.DATA / "images"   # ảnh minh họa dòng xe + CREDITS.csv (scripts/fetch_car_images.py)
FRONTEND_DIST = Path(os.environ.get("MLOPS_FRONTEND_DIST", ROOT / "frontend" / "dist"))
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Nạp sẵn mô hình (RandomForest ~240 MB) và dữ liệu gợi ý ở nền khi khởi động, để lần bấm đầu tiên không bị chờ.
    Lỗi (thiếu mô hình/dữ liệu) được bỏ qua ở đây; các endpoint vẫn báo lỗi rõ ràng khi được gọi."""
    def _warm():
        for fn in (get_predictor, get_recommender):
            try:
                fn()
            except Exception:
                pass
    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="Dự đoán giá xe", version="1.1", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
if IMAGES_DIR.is_dir():
    app.mount("/car-images", StaticFiles(directory=IMAGES_DIR), name="car-images")
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
            "brand_models": p.ref.get("brand_models", {}), "roles": {k: p.role(k) for k in p.models}}


@app.get("/api/metrics")
def metrics():
    """Chỉ số trên tập test của tỷ lệ chia đang được phục vụ (lấy từ lần chạy thực tế)."""
    p = get_predictor()
    return {"split": p.info["split"], "n_train": p.info["n_train"], "n_test": p.info["n_test"],
            "selection_criterion": p.info["selection_criterion"], "best_model": p.info["best_model"],
            "unit": p.info["price_unit"], "unit_note": p.info["price_unit_note"], "n_val": p.info.get("n_val"),
            "split_strategy": p.info.get("split_strategy"),
            "models": [{"model": k, "label": v["label"], "config_used": v.get("config_used"),
                        **{m: (None if v.get(m) is None else round(v[m], 3)) for m in ("cv_rmse", "val_RMSE", "val_RMSLE", "MAE", "RMSE", "RMSLE", "R2", "MAPE", "medAPE")},
                        "role": p.role(k)}
                       for k, v in p.info["models"].items()],
            "recommend_by_band": p.info.get("recommend_by_band")}


@app.get("/api/model-strengths")
def model_strengths():
    """Sai số từng mô hình theo nhóm xe (giá dự đoán, tuổi xe, số km) trên tập kiểm định.

    Chỉ ĐỌC reports/model_roles.csv do train_supervised.py ghi; không tính lại, không bịa số.
    `best` = mô hình có RMSLE thấp nhất trong nhóm; `close` = mô hình đứng nhì chỉ kém dưới 3% (khác biệt không đáng kể).
    """
    import csv
    path = C.REPORTS / "model_roles.csv"
    if not path.exists():
        return {"available": False, "reason": "Chưa có reports/model_roles.csv. Hãy huấn luyện: python src/train_supervised.py", "segments": []}
    groups: Dict[tuple, Dict[str, Any]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                key = (r["segment_type"], r["segment"])
                g = groups.setdefault(key, {"type": r["segment_type"], "segment": r["segment"], "n": int(float(r["n"])), "models": {}})
                g["models"][r["model"]] = {"rmsle": round(float(r["RMSLE"]), 4), "medape": round(float(r["medAPE"]), 2)}
            except (KeyError, ValueError):
                continue
    segs = []
    for g in groups.values():
        ranked = sorted(g["models"].items(), key=lambda kv: kv[1]["rmsle"])
        g["best"] = ranked[0][0] if ranked else None
        g["close"] = bool(len(ranked) > 1 and ranked[0][1]["rmsle"] > 0 and (ranked[1][1]["rmsle"] - ranked[0][1]["rmsle"]) / ranked[0][1]["rmsle"] < 0.03)
        segs.append(g)
    return {"available": bool(segs), "reason": None if segs else "File model_roles.csv rỗng.", "segments": segs}


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


@app.get("/api/mlflow")
def mlflow_runs():
    """Các lần huấn luyện và Model Registry đọc từ mlflow.db (chỉ đọc)."""
    from mlflow_view import mlflow_summary
    return mlflow_summary()


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


class RecommendRequest(BaseModel):
    """Bộ lọc gợi ý xe. Mọi trường đều tùy chọn; giá tính bằng VND."""
    budget_min: Optional[float] = Field(None, ge=0, le=1e12)
    budget_max: Optional[float] = Field(None, ge=0, le=1e12)
    brand: Optional[str] = Field(None, max_length=60)
    model: Optional[str] = Field(None, max_length=80)
    year_from: Optional[int] = Field(None, ge=1950, le=2100)
    fuel_type: Optional[str] = Field(None, max_length=30)
    transmission: Optional[str] = Field(None, max_length=30)
    body_type: Optional[str] = Field(None, max_length=30)
    method: Literal["knn", "content_based", "rule_based"] = "content_based"
    liked_ids: List[str] = Field(default_factory=list, max_length=5, description="Mã các xe người dùng thích (chỉ dùng với content_based).")
    limit: int = Field(12, ge=1, le=24)

    @model_validator(mode="after")
    def _budget_order(self):
        if self.budget_min is not None and self.budget_max is not None and self.budget_min > self.budget_max:
            raise ValueError("Ngân sách tối thiểu không được lớn hơn ngân sách tối đa.")
        return self


def get_recommender():
    try:
        return load_recommender(C.PROCESSED / "clean.csv", IMAGES_DIR)
    except DataNotAvailable as e:
        raise HTTPException(503, str(e))


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    """Gợi ý xe từ tin rao thật. Trả {items: [CarListing], method, total}; items rỗng nếu không có xe phù hợp."""
    f = req.model_dump(exclude={"method", "limit"})
    return get_recommender().recommend(f, req.method, req.limit, req.liked_ids)


@app.get("/api/cars/{car_id}")
def car_detail(car_id: str):
    """Chi tiết một tin rao (id dạng `bonbanh-123`), kèm similar_ids là các xe tương đồng (KNN)."""
    car = get_recommender().get(car_id)
    if car is None:
        raise HTTPException(404, "Không tìm thấy xe này trong dữ liệu.")
    return car


@app.get("/{path:path}", include_in_schema=False)
def spa_fallback(path: str):
    """Các đường dẫn của React Router (vd /dinh-gia): trả index.html. Đường dẫn /api/* không tồn tại vẫn trả 404 dạng JSON."""
    if path.startswith(("api/", "assets/", "static/", "car-images/")) or not (FRONTEND_DIST / "index.html").exists():
        raise HTTPException(404, "Không tìm thấy")
    f = (FRONTEND_DIST / path).resolve()
    if path and f.is_file() and FRONTEND_DIST.resolve() in f.parents:   # file tĩnh ở gốc dist (favicon...)
        return FileResponse(f)
    return FileResponse(FRONTEND_DIST / "index.html")
