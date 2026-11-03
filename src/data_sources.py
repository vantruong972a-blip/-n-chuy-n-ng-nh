"""Đọc dữ liệu THẬT (bonbanh, chotot) và đưa về cùng một lược đồ chuẩn (canonical schema).

Chỉ chuẩn hóa tên cột / đơn vị / cách viết; KHÔNG xóa dòng nào ở bước này (việc loại dòng nằm trong
preprocess.py và đều có ghi lý do). File gốc trong Data/raw/ không bao giờ bị sửa.

Lược đồ chuẩn (một dòng = một tin rao):
  source, source_id, brand, model, variant, body_type, year, mileage_km, engine_capacity (lít),
  transmission, fuel_type, drivetrain, seats, origin, province, color, price (VND)
"""
import json
import re
import unicodedata

import numpy as np
import pandas as pd

import config as C

CANON_COLS = ["source", "source_id", "brand", "model", "variant", "body_type", "year", "mileage_km",
              "engine_capacity", "transmission", "fuel_type", "drivetrain", "seats", "origin", "province",
              "color", "price"]

_BRAND_FIX = {"mercedes": "mercedes-benz", "mercedes benz": "mercedes-benz", "landrover": "land rover",
              "rolls": "rolls-royce", "alfa": "alfa romeo", "aston": "aston martin", "hãng khác": "other"}
_BODY = {"sedan": "sedan", "suv": "suv_crossover", "crossover": "suv_crossover", "suv / cross over": "suv_crossover",
         "hatchback": "hatchback", "pickup": "pickup", "pick-up (bán tải)": "pickup", "van": "van",
         "minivan (mpv)": "mpv", "coupe": "coupe", "coupe (2 cửa)": "coupe", "convertible": "convertible",
         "mui trần": "convertible", "wagon": "other", "kiểu dáng khác": "other"}
_TRANS = {"automatic": "automatic", "manual": "manual", "semi-automatic": "semi_automatic"}
_FUEL = {"gasoline": "gasoline", "diesel": "diesel", "hybrid": "hybrid", "động cơ hybrid": "hybrid",
         "electric": "electric"}

PROVINCES = """ha-noi tp-ho-chi-minh hai-phong da-nang can-tho an-giang ba-ria-vung-tau bac-giang bac-kan bac-lieu
bac-ninh ben-tre binh-dinh binh-duong binh-phuoc binh-thuan ca-mau cao-bang dak-lak dak-nong dien-bien dong-nai
dong-thap gia-lai ha-giang ha-nam ha-tinh hai-duong hau-giang hoa-binh hung-yen khanh-hoa kien-giang kon-tum
lai-chau lam-dong lang-son lao-cai long-an nam-dinh nghe-an ninh-binh ninh-thuan phu-tho phu-yen quang-binh
quang-nam quang-ngai quang-ninh quang-tri soc-trang son-la tay-ninh thai-binh thai-nguyen thanh-hoa
thua-thien-hue tien-giang tra-vinh tuyen-quang vinh-long vinh-phuc yen-bai""".split()
_PROV_SORTED = sorted(PROVINCES, key=len, reverse=True)  # khớp slug dài nhất trước


def norm_text(s) -> str | float:
    """Chữ thường, '-'/'_' -> khoảng trắng, gọn khoảng trắng. NaN giữ nguyên."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    t = re.sub(r"\s+", " ", re.sub(r"[-_]", " ", str(s).lower())).strip()
    return t if t else np.nan


def norm_brand(s):
    t = unicodedata.normalize("NFC", s).lower().strip() if isinstance(s, str) else ""
    t = re.sub(r"\s+", " ", t)
    if not t:
        return np.nan
    t = _BRAND_FIX.get(t, t)
    return t


def norm_model(s):
    t = norm_text(s)
    if isinstance(t, float):
        return t
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"\s+(class|series)$", "", t)           # "c class" -> "c", "3 series" -> "3"
    return "other" if t in ("dòng", "dong", "dòng khác") else t


def province_from_url(url) -> str | float:
    """Chotot đặt tỉnh/thành ở cuối slug địa điểm của URL, vd .../mua-ban-oto-cu-sdca1-quan-8-tp-ho-chi-minh/123.htm"""
    m = re.search(r"sdca1-([^/]+)/", str(url))
    if not m:
        return np.nan
    slug = m.group(1)
    for p in _PROV_SORTED:
        if slug == p or slug.endswith("-" + p):
            return p
    return np.nan


def _variant_bonbanh(name, brand, model, year):
    """Phiên bản = phần còn lại của tiêu đề sau khi bỏ hãng, dòng xe, năm, dung tích, hộp số (vd 'accent 1 4 ath 2018' -> 'ath')."""
    t = norm_text(name)
    if isinstance(t, float):
        return np.nan
    toks = t.split()
    if toks and toks[-1] == str(year):
        toks = toks[:-1]
    drop = set(str(norm_text(brand)).split()) | set(str(norm_text(model)).split()) | {"sh", "mt", "at", "cvt", "l"}
    rest, prev_num = [], False
    for w in toks:
        is_num = w.isdigit()
        if not is_num and prev_num and re.fullmatch(r"\d[a-z]+", w):   # '2 0e' đã tách thành '2','0e': bỏ chữ số thập phân của dung tích
            w = w[1:]
        prev_num = is_num
        if is_num or w in drop or re.fullmatch(r"\dx\d", w):          # bỏ số, từ đã có ở cột khác, '4x2/4x4' (cầu dẫn động)
            continue
        rest.append(w)
    return " ".join(rest) if rest else np.nan


def load_bonbanh(path=None) -> pd.DataFrame:
    path = path or C.RAW_FILES["bonbanh"]
    with open(path, encoding="utf-8") as f:
        raw = pd.DataFrame(json.load(f))
    out = pd.DataFrame({
        "source": "bonbanh",
        "source_id": raw["source_url"].astype(str).str.extract(r"(\d+)$")[0],
        "brand": raw["brand"].map(norm_brand),
        "model": raw["model"].map(norm_model),
        "variant": [_variant_bonbanh(n, b, m, y) for n, b, m, y in zip(raw["name"], raw["brand"], raw["model"], raw["year"])],
        "body_type": raw["type"].str.lower().map(_BODY),
        "year": pd.to_numeric(raw["year"], errors="coerce"),
        "mileage_km": pd.to_numeric(raw["km_driven"], errors="coerce"),
        "engine_capacity": pd.to_numeric(raw["engine_capacity"], errors="coerce"),
        "transmission": raw["transmission"].str.lower().map(_TRANS),
        "fuel_type": raw["fuels"].str.lower().map(_FUEL),
        "drivetrain": raw["wheel_drive"].str.lower(),
        "seats": pd.to_numeric(raw["seats"], errors="coerce"),
        "origin": raw["origin"].str.lower(),
        "province": np.nan,                       # bonbanh (bản crawl này) không có địa điểm
        "color": raw["external_color"].map(norm_text),
        "price": pd.to_numeric(raw["price"], errors="coerce"),
    })
    return out[CANON_COLS]


def load_chotot(path=None) -> pd.DataFrame:
    path = path or C.RAW_FILES["chotot"]
    raw = pd.read_csv(path)
    out = pd.DataFrame({
        "source": "chotot",
        "source_id": raw["url"].astype(str).str.extract(r"/(\d+)\.htm")[0],
        "brand": raw["brand"].map(norm_brand),
        "model": raw["model"].map(norm_model),
        "variant": np.nan,                        # chotot không có phiên bản
        "body_type": raw["type"].str.lower().map(_BODY),
        "year": pd.to_numeric(raw["date"], errors="coerce"),   # cột 'date' thực chất là NĂM SẢN XUẤT
        "mileage_km": pd.to_numeric(raw["km_driven"], errors="coerce"),
        "engine_capacity": pd.to_numeric(raw["engine_capacity"], errors="coerce"),
        "transmission": raw["transmission"].str.lower().map(_TRANS),   # giá trị lạ như '5' -> NaN
        "fuel_type": raw["fuel"].str.lower().map(_FUEL),
        "drivetrain": np.nan,
        "seats": pd.to_numeric(raw["seats"], errors="coerce"),
        "origin": raw["origin"].str.lower(),
        "province": raw["url"].map(province_from_url),
        "color": np.nan,
        "price": pd.to_numeric(raw["prices"], errors="coerce"),
    })
    return out[CANON_COLS]


def load_all() -> pd.DataFrame:
    """Gộp các nguồn có trong Data/raw/. Thiếu cả hai file -> báo lỗi rõ ràng, có hướng dẫn tải."""
    parts = []
    for name, loader in (("bonbanh", load_bonbanh), ("chotot", load_chotot)):
        if C.RAW_FILES[name].exists():
            parts.append(loader())
    if not parts:
        raise FileNotFoundError(
            f"Không thấy dữ liệu trong {C.RAW_DIR}. Chạy: python scripts/download_data.py")
    return pd.concat(parts, ignore_index=True)
