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

VALID = {"fueltype": "gas", "aspiration": "std", "carbody": "sedan", "drivewheel": "fwd",
         "enginesize": 130, "horsepower": 110, "curbweight": 2400, "citympg": 28, "highwaympg": 34}


@pytest.fixture(scope="session")
def valid_input():
    return dict(VALID)


@pytest.fixture(scope="session")
def trained():
    """Huấn luyện nhanh (lưới nhỏ, 2 tỷ lệ) một lần cho cả phiên kiểm thử."""
    import train_supervised
    return train_supervised.run([0.3, 0.2], fast=True, use_mlflow=False)
