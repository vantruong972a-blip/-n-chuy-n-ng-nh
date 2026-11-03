"""Feature engineering nằm TRONG Pipeline (FeatureEngineer) để huấn luyện và API dùng chung một đoạn code.

Đầu vào: các cột trong config.INPUT_FEATURES (cột thiếu/NaN được chấp nhận).
Đặc trưng phái sinh (chỉ từ đặc trưng đầu vào, KHÔNG dùng giá => không rò rỉ nhãn):
  car_age      = REFERENCE_YEAR - year           (tuổi xe tại thời điểm thu thập dữ liệu)
  km_per_year  = mileage_km / max(car_age, 1)
  km_reported  = 1 nếu có số km hợp lệ, 0 nếu không (bonbanh ghi 0 km cho >50% tin => coi là "chưa khai báo")
  condition    = new_like (km<=1000 và tuổi<=1) | used | km_unknown
  brand_model  = "hãng|dòng" (tên dòng xe như '3', 'c', 'x5' chỉ có nghĩa khi đi kèm hãng)
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C

NUM_FEATURES = ["year", "car_age", "mileage_km", "km_per_year", "km_reported", "engine_capacity", "seats"]
CAT_FEATURES = ["brand", "brand_model", "variant", "body_type", "transmission", "fuel_type", "drivetrain",
                "origin", "province", "condition"]
ENGINEERED = ["car_age", "km_per_year", "km_reported", "condition", "brand_model"]


def clean_mileage(km: pd.Series) -> pd.Series:
    """Số km không hợp lệ -> NaN: <=0 (bonbanh dùng 0 cho 'chưa khai báo'), >1.000.000 (vd 4294967295 do tràn số)."""
    km = pd.to_numeric(km, errors="coerce").astype(float)
    return km.where((km > 0) & (km <= C.MAX_KM))


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Không có tham số học từ dữ liệu (stateless) nên không thể rò rỉ thông tin từ test."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(list(X.columns)) if hasattr(X, "columns") else None
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in C.INPUT_FEATURES:
            if c not in X.columns:
                X[c] = np.nan
        out = pd.DataFrame(index=X.index)
        out["year"] = pd.to_numeric(X["year"], errors="coerce")
        out["car_age"] = (C.REFERENCE_YEAR - out["year"]).clip(lower=0)
        km = clean_mileage(X["mileage_km"])
        out["mileage_km"] = km
        out["km_per_year"] = (km / out["car_age"].clip(lower=1)).where(lambda s: s <= C.MAX_KM_PER_YEAR)
        out["km_reported"] = km.notna().astype(float)
        out["engine_capacity"] = pd.to_numeric(X["engine_capacity"], errors="coerce")
        out["seats"] = pd.to_numeric(X["seats"], errors="coerce")
        for c in ("brand", "variant", "body_type", "transmission", "fuel_type", "drivetrain", "origin", "province"):
            out[c] = X[c].astype("object").where(X[c].notna(), np.nan)
        out["brand_model"] = (X["brand"].astype(str) + "|" + X["model"].astype(str)).where(
            X["brand"].notna() & X["model"].notna(), np.nan)
        out["condition"] = np.select(
            [km.isna(), (km <= 1000) & (out["car_age"] <= 1)], ["km_unknown", "new_like"], default="used")
        return out[NUM_FEATURES + CAT_FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.array(NUM_FEATURES + CAT_FEATURES, dtype=object)


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Điền dung tích động cơ / số chỗ còn thiếu bằng trung vị của CHÍNH dòng xe (brand_model), nếu chưa gặp thì theo hãng,
    cuối cùng theo toàn bộ train. Học chỉ từ đặc trưng của train (không dùng giá). Cần vì người dùng API thường bỏ trống
    các trường này: điền trung vị toàn bộ dữ liệu (~2,0 lít) làm Vios 1.5 bị định giá như xe máy lớn."""

    def __init__(self, cols=("engine_capacity", "seats"), keys=("brand_model", "brand")):
        self.cols = cols
        self.keys = keys

    def fit(self, X, y=None):
        self.tables_ = {c: {k: X.groupby(k)[c].median().dropna().to_dict() for k in self.keys} for c in self.cols}
        self.global_ = {c: float(X[c].median()) for c in self.cols}
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            miss = X[c].isna()
            if not miss.any():
                continue
            v = pd.Series(np.nan, index=X.index)
            for k in self.keys:
                v = v.fillna(X[k].map(self.tables_[c][k]))
            X.loc[miss, c] = v[miss].fillna(self.global_[c])
        return X


def build_preprocessor(X=None, min_frequency: int = 20) -> Pipeline:
    """FeatureEngineer + GroupMedianImputer + ColumnTransformer. Chưa fit. `X` chỉ để tương thích chữ ký cũ (không dùng).

    Mã hóa one-hot gộp các giá trị hiếm (<min_frequency tin trong train) vào nhóm 'infrequent',
    và giá trị chưa từng thấy lúc train cũng rơi vào nhóm đó => API không lỗi với dòng xe mới.
    """
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                         ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                                              sparse_output=False))])
    ct = ColumnTransformer([("num", num_pipe, NUM_FEATURES), ("cat", cat_pipe, CAT_FEATURES)])
    return Pipeline([("fe", FeatureEngineer()), ("fill", GroupMedianImputer()), ("ct", ct)])
