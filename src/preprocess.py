"""Làm sạch dữ liệu, tiền xử lý (ColumnTransformer) và chia train/test.

Nguyên tắc: dữ liệu gốc Data/raw/car.csv không bị sửa; chia train/test bằng code
(get_split), bộ tiền xử lý chỉ fit trên tập train (nằm trong Pipeline cùng mô hình).
"""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C


def data_fingerprint(path=None) -> str:
    """Mã nhận diện phiên bản dữ liệu (SHA-256 rút gọn của file gốc)."""
    path = path or C.RAW_CSV
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def load_clean(path=None, with_target=True) -> pd.DataFrame:
    """Đọc CSV, chuẩn hóa tên cột, bỏ cột định danh, bỏ dòng trùng, bỏ dòng thiếu nhãn."""
    path = path or C.RAW_CSV
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.drop(columns=[c for c in C.DROP_COLS if c in df.columns])
    df = df.drop_duplicates()  # sau khi bỏ cột định danh mới phát hiện được dòng trùng thật
    if with_target:
        if C.TARGET not in df.columns:
            raise ValueError(f"Thiếu cột mục tiêu '{C.TARGET}' trong {path}")
        df[C.TARGET] = pd.to_numeric(df[C.TARGET], errors="coerce")
        df = df.dropna(subset=[C.TARGET])
        df = df[df[C.TARGET] > 0]
        if len(df) < C.MIN_ROWS:
            raise ValueError(f"Chỉ còn {len(df)} dòng hợp lệ (< {C.MIN_ROWS}); kiểm tra dữ liệu.")
    return df


def split_columns(X: pd.DataFrame):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return num, cat


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Chưa fit. Thứ tự cột đầu vào = thứ tự cột của X."""
    num, cat = split_columns(X)
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])


def get_split(test_size=C.SERVE_TEST_SIZE, random_state=C.RANDOM_STATE, df=None):
    """Chia (X_train, X_test, y_train, y_test) từ dữ liệu sạch. Chưa fit gì cả."""
    df = load_clean() if df is None else df
    X, y = df.drop(columns=[C.TARGET]), df[C.TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def main():
    """Ghi dữ liệu đã làm sạch (chưa scale) + thông tin chia dữ liệu để kiểm tra."""
    for d in (C.PROCESSED, C.SPLITS, C.MODELS, C.REPORTS):
        d.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    df.to_csv(C.PROCESSED / "clean.csv", index_label="row_id")
    rows = []
    for ts in C.TEST_SIZES:
        X_tr, X_te, _, _ = get_split(ts, df=df)
        rows.append({"test_size": ts, "train_pct": round((1 - ts) * 100), "test_pct": round(ts * 100),
                     "n_train": len(X_tr), "n_test": len(X_te)})
        if abs(ts - C.SERVE_TEST_SIZE) < 1e-9:
            pd.Series(X_tr.index, name="row_id").to_csv(C.SPLITS / "train_idx.csv", index=False)
            pd.Series(X_te.index, name="row_id").to_csv(C.SPLITS / "test_idx.csv", index=False)
    pd.DataFrame(rows).to_csv(C.SPLITS / "split_summary.csv", index=False)
    num, cat = split_columns(df.drop(columns=[C.TARGET]))
    info = {"data_file": str(C.RAW_CSV.name), "data_version": data_fingerprint(),
            "n_rows_clean": len(df), "n_features_raw": len(num) + len(cat),
            "numeric_features": num, "categorical_features": cat, "target": C.TARGET,
            "random_state": C.RANDOM_STATE, "splits": rows}
    (C.PROCESSED / "data_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
