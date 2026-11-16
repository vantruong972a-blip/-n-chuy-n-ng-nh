import os
import sys
import tempfile
from pathlib import Path

# Ghi kết quả vào thư mục tạm, không đụng models/ và reports/ thật.
_TMP = tempfile.mkdtemp(prefix="mlops_test_")
os.environ["MLOPS_WORKDIR"] = _TMP
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

VALID = {"brand": "toyota", "model": "vios", "year": 2018, "mileage_km": 60000, "transmission": "automatic",
         "fuel_type": "gasoline"}


@pytest.fixture(scope="session")
def valid_input():
    return dict(VALID)


@pytest.fixture(scope="session")
def cleaned():
    """Chạy bước làm sạch trên TOÀN BỘ dữ liệu thật trong Data/raw/ một lần (ghi vào thư mục tạm)."""
    import shutil
    import config as C
    import preprocess
    raw_ok = all(f.exists() for f in C.RAW_FILES.values())
    src = ROOT / "Data" / "processed"
    if not raw_ok and (src / "clean.csv").exists():      # bản clone không kèm dữ liệu thô: dùng bản đã làm sạch có sẵn
        shutil.copytree(src, C.PROCESSED, dirs_exist_ok=True)
    else:
        preprocess.main()
    return preprocess.load_clean()


@pytest.fixture(scope="session")
def trained(cleaned):
    """Huấn luyện nhanh (lưới nhỏ, mẫu 3000 dòng thật theo nhóm, 1 tỷ lệ) một lần cho cả phiên kiểm thử."""
    import train_supervised
    return train_supervised.run([0.2], fast=True, use_mlflow=False, max_rows=3000, n_iter=2)
