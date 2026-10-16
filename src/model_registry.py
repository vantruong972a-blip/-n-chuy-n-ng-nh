"""Danh sách mô hình hồi quy và lưới siêu tham số. Muốn thêm mô hình: thêm 1 mục vào get_specs()."""
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import config as C

LABELS = {"linear_regression": "Linear Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}


def get_specs(random_state=C.RANDOM_STATE, fast=False):
    """Trả về {tên: (estimator, param_grid)}. Tham số lưới có tiền tố 'model__' (dùng trong Pipeline)."""
    specs = {
        "linear_regression": (LinearRegression(), {"model__fit_intercept": [True, False]}),
        "random_forest": (
            RandomForestRegressor(random_state=random_state, n_jobs=-1),
            {"model__n_estimators": [100] if fast else [200, 400],
             "model__max_depth": [None, 8] if fast else [None, 8, 16],
             "model__min_samples_leaf": [1] if fast else [1, 3]}),
    }
    try:
        from xgboost import XGBRegressor
        specs["xgboost"] = (
            XGBRegressor(random_state=random_state, subsample=0.9, n_jobs=1),
            {"model__n_estimators": [100] if fast else [200, 400],
             "model__max_depth": [3] if fast else [3, 5],
             "model__learning_rate": [0.1] if fast else [0.05, 0.1]})
    except ImportError:
        warnings.warn("Chưa cài xgboost -> bỏ qua XGBoost. Chạy: pip install xgboost")
    return specs
