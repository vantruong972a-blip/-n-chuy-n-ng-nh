"""Nạp mô hình (Pipeline = tiền xử lý + mô hình đã fit lúc train) và dự đoán.

Không bao giờ fit lại tiền xử lý ở đây. Cột đầu vào luôn được sắp theo đúng thứ tự lúc train.
"""
import json
import uuid

import joblib
import numpy as np
import pandas as pd

import config as C
from model_registry import LABELS


class ModelNotAvailable(Exception):
    """Thiếu mô hình/bộ tiền xử lý (chưa huấn luyện)."""


class InvalidInput(Exception):
    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = errors


class Predictor:
    def __init__(self):
        info_p, ref_p = C.MODELS / "model_info.json", C.MODELS / "reference_stats.json"
        if not info_p.exists() or not ref_p.exists():
            raise ModelNotAvailable("Thiếu models/model_info.json hoặc reference_stats.json. "
                                    "Hãy chạy: python src/train_supervised.py --test-size 0.2")
        self.info = json.loads(info_p.read_text(encoding="utf-8"))
        self.ref = json.loads(ref_p.read_text(encoding="utf-8"))
        self.features = self.info["features"]
        self.models = {}
        for name in self.info["models"]:
            p = C.MODELS / f"{name}.joblib"
            if p.exists():
                self.models[name] = joblib.load(p)
        if not self.models:
            raise ModelNotAvailable("Không có file mô hình .joblib trong models/. Hãy huấn luyện trước.")
        self.default_model = self.info["best_model"] if self.info["best_model"] in self.models else next(iter(self.models))

    @property
    def labels(self):
        return {k: LABELS.get(k, k) for k in self.models}

    def feature_meta(self):
        meta = []
        for c in self.features:
            if c in self.ref["numeric"]:
                r = self.ref["numeric"][c]
                meta.append({"name": c, "type": "number", "min": r["min"], "max": r["max"], "median": r["median"]})
            else:
                r = self.ref["categorical"][c]
                meta.append({"name": c, "type": "category", "options": sorted(r["freq"]), "default": r["default"]})
        return meta

    def validate(self, values: dict) -> pd.DataFrame:
        errs, row = [], {}
        unknown = [k for k in values if k not in self.features]
        if unknown:
            errs.append(f"Trường không hợp lệ: {unknown}. Các trường hợp lệ: {self.features}")
        for c in self.features:
            v = values.get(c)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                errs.append(f"Thiếu giá trị cho '{c}'")
                continue
            if c in self.ref["numeric"]:
                r = self.ref["numeric"][c]
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    errs.append(f"'{c}' phải là số, nhận được {v!r}")
                    continue
                if not np.isfinite(x):
                    errs.append(f"'{c}' phải là số hữu hạn")
                    continue
                span = max(r["max"] - r["min"], 1e-9)
                lo, hi = r["min"] - 0.5 * span, r["max"] + 0.5 * span
                if r["min"] >= 0:
                    lo = max(lo, 0.0)
                if not lo <= x <= hi:
                    errs.append(f"'{c}'={x:g} nằm ngoài phạm vi hợp lý [{lo:g}, {hi:g}] (dữ liệu train: {r['min']:g}–{r['max']:g})")
                    continue
                row[c] = x
            else:
                s = str(v)
                opts = sorted(self.ref["categorical"][c]["freq"])
                if s not in opts:
                    errs.append(f"'{c}'={s!r} không thuộc {opts}")
                    continue
                row[c] = s
        if errs:
            raise InvalidInput(errs)
        return pd.DataFrame([row], columns=self.features)  # đúng thứ tự cột lúc train

    def predict(self, values: dict, model: str | None = None):
        X = self.validate(values)
        if model is not None and model not in self.models:
            raise InvalidInput([f"Mô hình '{model}' không tồn tại. Có sẵn: {list(self.models)}"])
        name = model or self.default_model
        allp = {k: float(m.predict(X)[0]) for k, m in self.models.items()}  # cùng 1 đầu vào cho mọi mô hình
        return {"prediction_id": uuid.uuid4().hex[:12], "model": name, "model_label": LABELS.get(name, name),
                "price": allp[name], "unit": self.info["price_unit"], "unit_note": self.info["price_unit_note"],
                "model_version": self.info["model_version"], "data_version": self.info["data_version"],
                "all": [{"model": k, "label": LABELS.get(k, k), "price": v} for k, v in allp.items()],
                "inputs": X.iloc[0].to_dict()}
