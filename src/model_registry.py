"""Danh sách mô hình hồi quy, giá trị mặc định (baseline) và không gian siêu tham số.

Muốn thêm mô hình: thêm 1 mục vào get_specs(). Tham số trong lưới có tiền tố 'model__regressor__' vì mô hình nằm trong
TransformedTargetRegressor (huấn luyện trên log1p(giá) rồi đổi ngược về VND), tiền tố 'pre__ct__cat__oh__' cho bộ mã hóa.
"""
import warnings

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import config as C
from features import build_preprocessor

LABELS = {"linear_regression": "Linear Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}
MF = "pre__ct__cat__oh__min_frequency"


def make_pipeline(estimator, min_frequency=20, log_target=True) -> Pipeline:
    """Pipeline(tiền xử lý + feature engineering + mô hình). log_target=True: dự đoán log1p(giá) rồi expm1 (giá luôn > 0)."""
    reg = TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1) if log_target else estimator
    return Pipeline([("pre", build_preprocessor(min_frequency=min_frequency)), ("model", reg)])


def get_specs(random_state=C.RANDOM_STATE, fast=False):
    """{tên: (estimator_mặc_định, lưới_tham_số)}. Estimator mặc định = baseline (tham số mặc định của thư viện)."""
    R = "model__regressor__"
    specs = {
        "linear_regression": (LinearRegression(), {MF: [10, 30, 100]}),   # LR gần như không có siêu tham số: chỉ chỉnh mức gộp nhóm hiếm
        "random_forest": (
            RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
            {R + "n_estimators": [60] if fast else [150, 250],
             R + "max_depth": [12] if fast else [12, 20, 30, None],
             R + "min_samples_leaf": [1, 3] if fast else [1, 2, 5],
             R + "max_features": [0.5] if fast else [0.3, 0.5, 1.0]}),
    }
    try:
        from xgboost import XGBRegressor
        specs["xgboost"] = (
            XGBRegressor(n_estimators=100, random_state=random_state, tree_method="hist", n_jobs=2),
            {R + "n_estimators": [80] if fast else [300, 600, 900],
             R + "max_depth": [4] if fast else [4, 6, 8],
             R + "learning_rate": [0.1] if fast else [0.03, 0.06, 0.1],
             R + "subsample": [0.8] if fast else [0.7, 0.9],
             R + "colsample_bytree": [0.8] if fast else [0.6, 0.9],
             R + "min_child_weight": [1] if fast else [1, 5]})
    except ImportError:
        warnings.warn("Chưa cài xgboost -> bỏ qua XGBoost. Chạy: pip install xgboost")
    return specs
