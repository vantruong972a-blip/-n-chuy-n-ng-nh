"""Cấu hình chung: đường dẫn, tham số thí nghiệm.

Có thể đổi thư mục xuất kết quả bằng biến môi trường MLOPS_WORKDIR
(dùng cho kiểm thử, để không ghi đè models/ và reports/ thật).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
RAW_CSV = DATA / "raw" / "car.csv"
UNLABELED_CSV = DATA / "external" / "car_unlabeled.csv"

_work = os.environ.get("MLOPS_WORKDIR")
_base = Path(_work) if _work else ROOT
PROCESSED = (_base / "Data" / "processed") if _work else DATA / "processed"
SPLITS = (_base / "Data" / "splits") if _work else DATA / "splits"
MODELS = _base / "models"
REPORTS = _base / "reports"
LOGS = _base / "logs"
MLRUNS = _base / "mlruns"
MLFLOW_DB = _base / "mlflow.db"
TARGET = "price"
RANDOM_STATE = 42

# Tỷ lệ test cần thử: 40% (60/40), 30% (70/30), 20% (80/20), 10% (90/10)
TEST_SIZES = [0.4, 0.3, 0.2, 0.1]
# Tỷ lệ dùng để lưu mô hình cho API
SERVE_TEST_SIZE = 0.2
CV_FOLDS = 5

# Cột định danh/không dùng làm đặc trưng (bỏ nếu tồn tại)
DROP_COLS = ["car_id", "carname"]

# Đơn vị giá: dataset hiện tại là dữ liệu mô phỏng nên đơn vị chỉ là giả định.
# Khi thay bằng dữ liệu thật, hãy sửa lại cho đúng nguồn (hoặc đặt biến môi trường).
PRICE_UNIT = os.environ.get("MLOPS_PRICE_UNIT", "USD")
PRICE_UNIT_NOTE = os.environ.get(
    "MLOPS_PRICE_UNIT_NOTE", "Đơn vị giả định (dữ liệu mô phỏng); cần xác nhận theo dataset thật.")

MLFLOW_EXPERIMENT = "car-price-prediction"
MIN_ROWS = 30  # số dòng tối thiểu sau làm sạch để chạy thí nghiệm
