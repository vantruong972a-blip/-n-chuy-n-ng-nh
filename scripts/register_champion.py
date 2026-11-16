"""Đăng ký mô hình ĐANG PHỤC VỤ API (models/model_info.json) vào MLflow Model Registry, bí danh 'champion'.

Dùng khi bạn đã huấn luyện từ trước (mlflow.db chưa có Registry). Huấn luyện mới tự làm bước này.
    python scripts/register_champion.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
import tracking  # noqa: E402


def main():
    info = json.loads((C.MODELS / "model_info.json").read_text(encoding="utf-8"))
    if not tracking.setup(True):
        raise SystemExit("Không dùng được MLflow (chưa cài mlflow?).")
    reg = tracking.register_champion(info["model_version"], info["best_model"], True)
    if not reg:
        raise SystemExit(f"Không tìm thấy lần chạy MLflow cho bản {info['model_version']} / {info['best_model']}. "
                         "Hãy huấn luyện lại: python src\\train_supervised.py")
    print(f"Đã đăng ký {reg['name']} v{reg['version']} (bí danh '{reg['alias']}') cho bản {info['model_version']}")


if __name__ == "__main__":
    main()
