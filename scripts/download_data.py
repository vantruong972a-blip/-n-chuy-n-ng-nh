"""Tải lại dữ liệu thật vào Data/raw/ từ commit cố định của repo Hyprnx/used-cars-prices-prediction và kiểm tra SHA-256.

Dùng khi Data/raw/ bị thiếu/hỏng hoặc khi cài trên máy mới mà không copy kèm dữ liệu:
    python scripts/download_data.py          # chỉ tải file thiếu/sai checksum
    python scripts/download_data.py --force  # tải lại tất cả
Nếu máy bạn chặn raw.githubusercontent.com, hãy tải 2 file thủ công (đường dẫn ghi trong Data/raw/SOURCES.md)
rồi đặt vào Data/raw/; script sẽ kiểm tra checksum.
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

COMMIT = "d0ee2b5afae0b4bf008e59f46299309f5154cb9a"
BASE = f"https://raw.githubusercontent.com/Hyprnx/used-cars-prices-prediction/{COMMIT}/data"
FILES = {  # tên file -> (đường dẫn trong repo gốc, SHA-256)
    "bonbanh_used_cars.json": ("bonbanh/bonbanh_used_cars.json",
                               "29ff9cafb98b2e7edc0289cb86b7ab24f59a73634b16356b677aef9a4d87d247"),
    "chotot_used_cars.csv": ("chotot/chotot_used_cars.csv",
                             "a1ee7c380057d518d4101873f1d6fcae2bfbcc81ad21993536ee6167bbc135e0"),
}
DEST = Path(__file__).resolve().parents[1] / "Data" / "raw"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, (rel, want) in FILES.items():
        p = DEST / name
        if p.exists() and sha256(p) == want and not a.force:
            print(f"OK       {name} (checksum khớp)")
            continue
        url = f"{BASE}/{rel}"
        print(f"Tải      {url}")
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                p.write_bytes(r.read())
        except Exception as e:  # noqa: BLE001
            print(f"LỖI      không tải được {name}: {e}\n         Hãy tải thủ công về {p}", file=sys.stderr)
            ok = False
            continue
        got = sha256(p)
        print(("OK       " if got == want else "SAI CHECKSUM ") + f"{name} sha256={got[:16]}...")
        ok &= got == want
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
