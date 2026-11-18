"""Nạp mô hình (Pipeline = feature engineering + tiền xử lý + mô hình đã fit lúc train) và dự đoán.

Không bao giờ fit lại tiền xử lý ở đây. Chỉ bắt buộc: brand, model, year; các trường còn lại tùy chọn
(để trống = thiếu => bộ impute học từ train xử lý, giống lúc huấn luyện).
"""
import json
import math
import unicodedata
import uuid

import joblib
import numpy as np
import pandas as pd

import config as C
import evaluation as E
from model_registry import LABELS

NUMERIC_RULES = {  # (min, max, kiểu nguyên?)  -- phạm vi HỢP LỆ tuyệt đối, không phải phạm vi quan sát được
    "year": (C.YEAR_MIN, C.YEAR_MAX, True),
    "mileage_km": (0, C.MAX_KM, False),
    "engine_capacity": (0.6, 8.0, False),
    "seats": (2, 16, True),
}
FREE_TEXT = {"variant"}   # giá trị chưa gặp chỉ cảnh báo
FIELD_LABELS = {"brand": "Hãng xe", "model": "Dòng xe", "variant": "Phiên bản", "body_type": "Kiểu dáng", "year": "Năm sản xuất",
                "mileage_km": "Số km đã đi", "engine_capacity": "Dung tích động cơ (lít)", "transmission": "Hộp số",
                "fuel_type": "Nhiên liệu", "drivetrain": "Dẫn động", "seats": "Số chỗ", "origin": "Xuất xứ", "province": "Nơi rao bán (tỉnh/thành)"}


class ModelNotAvailable(Exception):
    """Thiếu mô hình/bộ tiền xử lý (chưa huấn luyện)."""


class InvalidInput(Exception):
    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = errors


def _norm(s) -> str:
    return " ".join(unicodedata.normalize("NFC", str(s)).lower().replace("_", " ").split())


def _is_blank(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "") or (isinstance(v, float) and math.isnan(v))


class Predictor:
    def __init__(self):
        info_p, ref_p = C.MODELS / "model_info.json", C.MODELS / "reference_stats.json"
        if not info_p.exists() or not ref_p.exists():
            raise ModelNotAvailable("Thiếu models/model_info.json hoặc reference_stats.json. "
                                    "Hãy chạy: python src/train_supervised.py")
        self.info = json.loads(info_p.read_text(encoding="utf-8"))
        self.ref = json.loads(ref_p.read_text(encoding="utf-8"))
        self.features = self.info["features"]
        if self.features != C.INPUT_FEATURES:
            raise ModelNotAvailable("Lược đồ đặc trưng của mô hình đã lưu khác config.INPUT_FEATURES; hãy huấn luyện lại.")
        self.required = self.info.get("required_features", C.REQUIRED_FEATURES)
        self.models = {}
        for name in self.info["models"]:
            p = C.MODELS / f"{name}.joblib"
            if p.exists():
                self.models[name] = joblib.load(p)
        if not self.models:
            raise ModelNotAvailable("Không có file mô hình .joblib trong models/. Hãy huấn luyện trước.")
        self.default_model = self.info["best_model"] if self.info["best_model"] in self.models else next(iter(self.models))
        self._brand_models = self.ref.get("brand_models", {})

    def role(self, name):
        """Vai trò đã đo của mô hình (từ model_info.json); bản mô hình cũ chưa có thì trả None."""
        r = (self.info.get("roles") or {}).get(name)
        return None if not r else {"title": r.get("title"), "summary": r.get("summary"), "wins_rmsle": r.get("wins_rmsle"),
                                   "wins_medAPE": r.get("wins_medAPE"), "n_segments": r.get("n_segments")}

    def recommend(self, price_default: float):
        """Mô hình nên tin nhất cho dải giá dự đoán hiện tại (đo trên validation); None nếu bản mô hình chưa có thông tin."""
        for b in self.info.get("recommend_by_band") or []:
            if price_default >= b["lo"] and (b["hi"] is None or price_default < b["hi"]):
                m = b["model"]
                return {"model": m, "label": LABELS.get(m, m), "band": b["label"], "n": b["n"], "rmsle": b.get("rmsle"),
                        "is_default": m == self.default_model}
        return None

    @property
    def labels(self):
        return {k: LABELS.get(k, k) for k in self.models}

    def feature_meta(self):
        meta = []
        for c in self.features:
            base = {"name": c, "label": FIELD_LABELS.get(c, c), "required": c in self.required}
            if c in self.ref["numeric"]:
                r = self.ref["numeric"][c]
                lo, hi, is_int = NUMERIC_RULES.get(c, (None, None, False))
                meta.append({**base, "type": "number", "integer": is_int, "min": lo, "max": hi,
                             "train_min": r["min"], "train_max": r["max"], "median": r["median"]})
            else:
                r = self.ref["categorical"][c]
                opts = sorted(r["freq"], key=lambda k: -r["freq"][k])
                meta.append({**base, "type": "category", "free_text": c in FREE_TEXT,
                             "options": opts if len(opts) <= 120 else opts[:120], "default": r["default"]})
        return meta

    # ------------------------------------------------------------------ kiểm tra đầu vào
    def validate(self, values: dict):
        """Trả (DataFrame 1 dòng đúng thứ tự cột lúc train, warnings, defaults_used). Lỗi -> InvalidInput."""
        if not isinstance(values, dict):
            raise InvalidInput(["'values' phải là một object {tên_trường: giá_trị}"])
        errs, warns, row, omitted = [], [], {}, []
        unknown = [k for k in values if k not in self.features]
        if unknown:
            errs.append(f"Trường không hợp lệ: {unknown}. Các trường hợp lệ: {self.features}")
        for c in self.features:
            v = values.get(c)
            if _is_blank(v):
                if c in self.required:
                    errs.append(f"Thiếu giá trị bắt buộc cho '{c}'")
                else:
                    omitted.append(c)
                    row[c] = np.nan
                continue
            if c in NUMERIC_RULES:
                lo, hi, is_int = NUMERIC_RULES[c]
                if isinstance(v, bool):
                    errs.append(f"'{c}' phải là số, nhận được {v!r}")
                    continue
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    errs.append(f"'{c}' phải là số, nhận được {v!r}")
                    continue
                if not np.isfinite(x):
                    errs.append(f"'{c}' phải là số hữu hạn")
                    continue
                if not lo <= x <= hi:
                    errs.append(f"'{c}'={x:g} nằm ngoài phạm vi hợp lệ [{lo:g}, {hi:g}]")
                    continue
                if is_int and x != int(x):
                    errs.append(f"'{c}' phải là số nguyên, nhận được {x:g}")
                    continue
                if c == "mileage_km" and x == 0:
                    warns.append("mileage_km=0 được coi là 'chưa khai báo số km' (giống cách xử lý dữ liệu huấn luyện).")
                    row[c] = np.nan
                    continue
                tr = self.ref["numeric"].get(c)
                if tr and not (tr["min"] <= x <= tr["max"]):
                    warns.append(f"'{c}'={x:g} nằm ngoài khoảng dữ liệu huấn luyện ({tr['min']:g}–{tr['max']:g}); dự đoán kém tin cậy.")
                row[c] = x
            else:
                if not isinstance(v, str):
                    errs.append(f"'{c}' phải là chuỗi, nhận được {v!r}")
                    continue
                s = _norm(v)
                known = self.ref["categorical"][c]["freq"]
                if c == "model" and s not in known:
                    pass
                elif s not in known:
                    if c in FREE_TEXT:
                        warns.append(f"'{c}'={s!r} chưa từng thấy khi huấn luyện; được xử lý như nhóm hiếm.")
                    else:
                        errs.append(f"'{c}'={s!r} không thuộc các giá trị đã biết ({sorted(known, key=lambda k: -known[k])[:12]}...)"
                                    if len(known) > 12 else f"'{c}'={s!r} không thuộc {sorted(known)}")
                        continue
                row[c] = s
        if "model" in row and "brand" in row and isinstance(row.get("model"), str) and isinstance(row.get("brand"), str):
            if row["brand"] in self._brand_models and row["model"] not in self._brand_models[row["brand"]]:
                warns.append(f"Dòng xe {row['model']!r} chưa gặp (hoặc rất hiếm) với hãng {row['brand']!r} khi huấn luyện; "
                             f"mô hình sẽ dựa chủ yếu vào hãng và các thuộc tính còn lại.")
        if errs:
            raise InvalidInput(errs)
        return pd.DataFrame([row], columns=self.features), warns, omitted

    def predict(self, values: dict, model: str | None = None):
        X, warns, omitted = self.validate(values)
        if model is not None and model not in self.models:
            raise InvalidInput([f"Mô hình '{model}' không tồn tại. Có sẵn: {list(self.models)}"])
        name = model or self.default_model
        allp = {k: float(m.predict(X)[0]) for k, m in self.models.items()}  # cùng 1 đầu vào cho mọi mô hình
        inputs = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in X.iloc[0].to_dict().items()}
        def rng(k):
            return E.apply_band(self.info["models"].get(k, {}).get("error_bands"), allp[k], self.info.get("error_band_level", E.BAND_LEVEL))
        return {"prediction_id": uuid.uuid4().hex[:12], "model": name, "range": rng(name), "model_label": LABELS.get(name, name),
                "price": allp[name], "unit": self.info["price_unit"], "unit_note": self.info["price_unit_note"],
                "model_version": self.info["model_version"], "data_version": self.info["data_version"],
                "all": [{"model": k, "label": LABELS.get(k, k), "price": v, "range": rng(k), "role": self.role(k)} for k, v in allp.items()],
                "recommended": self.recommend(allp[self.default_model]) if self.default_model in allp else None,
                "warnings": warns, "defaults_used": omitted, "inputs": inputs}
