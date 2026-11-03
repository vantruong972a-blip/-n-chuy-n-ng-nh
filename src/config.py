"""Cấu hình chung: đường dẫn, tham số thí nghiệm.

Có thể đổi thư mục xuất kết quả bằng biến môi trường MLOPS_WORKDIR
(dùng cho kiểm thử, để không ghi đè models/ và reports/ thật).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
RAW_DIR = Path(os.environ.get("MLOPS_RAW_DIR", DATA / "raw"))
DEMO_DIR = DATA / "demo"            # dữ liệu MÔ PHỎNG cũ (giữ lại để đối chiếu, không dùng để huấn luyện)

# Dữ liệu thật (tiếng Việt, đơn vị VND). Nguồn + checksum: Data/raw/SOURCES.md, Data/raw/CHECKSUMS.sha256
RAW_FILES = {"bonbanh": RAW_DIR / "bonbanh_used_cars.json", "chotot": RAW_DIR / "chotot_used_cars.csv"}

_work = os.environ.get("MLOPS_WORKDIR")
_base = Path(_work) if _work else ROOT
PROCESSED = (_base / "Data" / "processed") if _work else DATA / "processed"
SPLITS = (_base / "Data" / "splits") if _work else DATA / "splits"
MODELS = _base / "models"
REPORTS = _base / "reports"
LOGS = _base / "logs"
MLRUNS = _base / "mlruns"
MLFLOW_DB = _base / "mlflow.db"

TARGET = "price"                    # giá niêm yết, đơn vị VND
RANDOM_STATE = 42
# Năm tham chiếu để tính car_age. Dữ liệu được thu thập năm 2022 (năm xe lớn nhất = 2022), nên
# mô hình định giá theo mặt bằng giá 2022. Đây là hằng số của DỮ LIỆU, không phải ngày hiện tại.
REFERENCE_YEAR = 2022

# Tỷ lệ test cần thử: 40% (60/40), 30% (70/30), 20% (80/20), 10% (90/10)
TEST_SIZES = [0.4, 0.3, 0.2, 0.1]
SERVE_TEST_SIZE = 0.2               # tỷ lệ dùng để lưu mô hình cho API
VAL_FRACTION = 0.25                 # phần của (train+val) dùng làm validation => 80/20 => 60/20/20
CV_FOLDS = 3                        # GroupKFold khi tinh chỉnh (chỉ trên tập train)
TUNE_ITER = 10                      # số tổ hợp thử của RandomizedSearchCV cho RF/XGBoost

# Đặc trưng đầu vào của mô hình (người dùng/API nhập). Các cột phái sinh (car_age, km_per_year, ...)
# được tính TRONG Pipeline (features.FeatureEngineer) để huấn luyện và phục vụ giống hệt nhau.
INPUT_FEATURES = ["brand", "model", "variant", "body_type", "year", "mileage_km", "engine_capacity",
                  "transmission", "fuel_type", "drivetrain", "seats", "origin", "province"]
REQUIRED_FEATURES = ["brand", "model", "year"]   # các trường bắt buộc khi gọi API; còn lại tùy chọn
# Cột chỉ để truy vết/nhóm, KHÔNG đưa vào mô hình
META_COLS = ["source", "source_id", "group_id"]

PRICE_UNIT = "VND"
PRICE_UNIT_NOTE = ("Giá niêm yết (asking price) trên bonbanh.com/chotot.com, thu thập khoảng năm 2022; "
                   "không phải giá giao dịch thực tế và không phản ánh mặt bằng giá hiện tại.")
# Ngưỡng làm sạch giá (xem preprocess.py: mỗi quy tắc đều ghi lý do vào rejected_rows.csv)
PRICE_MIN_VND = 10_000_000          # < 10 triệu: không phải giá xe ô tô (lỗi đơn vị/nhập liệu, vd 0, 5, 7, 55)
PRICE_MAX_VND = 100_000_000_000     # > 100 tỷ: vượt xa xe đắt nhất trong dữ liệu (Rolls-Royce Phantom 68 tỷ)
PRICE_RATIO_LIMIT = 15.0            # lệch >15 lần so với trung vị cùng (hãng, dòng xe) => nghi lỗi nhập liệu
PRICE_RATIO_MIN_GROUP = 8           # chỉ kiểm tra tỷ lệ khi nhóm có >= 8 tin
YEAR_MIN, YEAR_MAX = 1980, 2022
MAX_KM = 1_000_000                  # > 1 triệu km (vd 4294967295 = tràn số) => km không hợp lệ (đặt thiếu)
MAX_KM_PER_YEAR = 100_000

MLFLOW_EXPERIMENT = "car-price-prediction"
MIN_ROWS = 30  # số dòng tối thiểu sau làm sạch để chạy thí nghiệm
