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
from features import build_linear_preprocessor, build_preprocessor

LABELS = {"linear_regression": "Linear Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}
MF = "pre__ct__cat__oh__min_frequency"


# Mỗi thuật toán dùng bộ đặc trưng hợp với bản chất của nó (cùng đầu vào thô nên API không đổi):
#  - linear_regression: đặc trưng KHẤU HAO (tuổi, tuổi², log km, độ dốc riêng từng hãng) vì mô hình tuyến tính chỉ học được quy luật cộng/nhân;
#  - random_forest, xgboost: cây tự tìm tương tác/phi tuyến nên chỉ cần số gốc + one-hot.
LINEAR_MODELS = {"linear_regression"}

# Vai trò đo được trên dữ liệu thật (xem reports/model_roles.md, sinh lại mỗi lần huấn luyện):
ROLES = {
    "xgboost": {"title": "Định giá chính", "summary": "Hạn chế những lần sai lệch lớn: sai số log và RMSE thấp nhất ở hầu hết phân khúc, "
                                                      "nhất là xe đắt, xe đời cũ và xe giá rẻ."},
    "random_forest": {"title": "Đối chứng ổn định", "summary": "Trung bình nhiều cây độc lập nên ít nhạy với nhiễu; sai số trung vị ngang "
                                                              "XGBoost nhưng thỉnh thoảng sai lớn hơn ở xe đắt. Dùng để đối chiếu giá XGBoost."},
    "linear_regression": {"title": "Quy luật khấu hao", "summary": "Giải thích được: mỗi năm tuổi, mỗi km làm giá giảm bao nhiêu % theo "
                                                                 "từng hãng; dùng để kiểm tra độ hợp lý, kém chính xác hơn hai mô hình cây."},
}


def make_pipeline(estimator, min_frequency=20, log_target=True, name=None) -> Pipeline:
    """Pipeline(tiền xử lý + feature engineering + mô hình). log_target=True: dự đoán log1p(giá) rồi expm1 (giá luôn > 0).
    `name` chọn bộ đặc trưng riêng của thuật toán (linear_regression dùng đặc trưng khấu hao); bỏ trống = bộ chung."""
    reg = TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1) if log_target else estimator
    pre = build_linear_preprocessor(min_frequency=min_frequency) if name in LINEAR_MODELS else build_preprocessor(min_frequency=min_frequency)
    return Pipeline([("pre", pre), ("model", reg)])


def get_specs(random_state=C.RANDOM_STATE, fast=False):
    """{tên: (estimator_mặc_định, lưới_tham_số)}. Estimator mặc định = baseline (tham số mặc định của thư viện)."""
    R = "model__regressor__"
    specs = {
        "linear_regression": (LinearRegression(), {MF: [10, 30, 100]}),   # LR gần như không có siêu tham số: chỉ chỉnh mức gộp nhóm hiếm
        # (đã thử Ridge: phạt L2 làm yếu độ dốc khấu hao riêng từng hãng nên sai số tăng => giữ bình phương tối thiểu thường)
        "random_forest": (
            RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
            # Lưới giữ file mô hình < ~150 MB (bản 300 cây, lá 1 mẫu nặng ~600 MB): <= 150 cây, mỗi lá tối thiểu 2 mẫu.
            {R + "n_estimators": [60] if fast else [120, 150],
             R + "max_depth": [12] if fast else [20, 30],
             R + "min_samples_leaf": [2, 3] if fast else [2, 3, 5],
             R + "max_features": [0.5] if fast else [0.5, 0.7, 1.0]}),
    }
    try:
        from xgboost import XGBRegressor
        specs["xgboost"] = (
            XGBRegressor(n_estimators=100, random_state=random_state, tree_method="hist", n_jobs=2),
            {R + "n_estimators": [80] if fast else [500, 800, 1100],
             R + "max_depth": [4] if fast else [6, 8, 10],
             R + "learning_rate": [0.1] if fast else [0.03, 0.05, 0.08],
             R + "subsample": [0.8] if fast else [0.8, 0.9],
             R + "colsample_bytree": [0.8] if fast else [0.7, 0.9],
             R + "colsample_bynode": [0.8] if fast else [0.5, 0.8],
             R + "min_child_weight": [1] if fast else [1, 3],
             R + "reg_lambda": [1.0] if fast else [1.0, 5.0]})
    except ImportError:
        warnings.warn("Chưa cài xgboost -> bỏ qua XGBoost. Chạy: pip install xgboost")
    return specs
