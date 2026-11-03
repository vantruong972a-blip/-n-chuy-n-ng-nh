"""Đo độ nhạy của chỉ số test với cách định nghĩa nhóm tin rao (group_id) khi chia dữ liệu.
So sánh: nhóm 'chặt' (gồm cả km => tin đăng lại có km khác nhau bị coi là xe khác, dễ rò rỉ) và nhóm 'bảo thủ' (không gồm km, là mặc định).
Mô hình: Random Forest/XGBoost tham số mặc định, 1 lần chia 80/20. Ghi reports/group_sensitivity.csv.   python scripts/group_sensitivity.py
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402
import evaluation as E  # noqa: E402
from model_registry import get_specs, make_pipeline  # noqa: E402
from preprocess import get_split3, load_clean  # noqa: E402

LOOSE = ["source", "brand", "model", "variant", "year", "engine_capacity", "transmission", "fuel_type", "seats", "color", "province"]


def main():
    df = load_clean()
    rows = []
    for name, extra in (("chặt (gồm km)", ["mileage_km"]), ("bảo thủ (không km) - mặc định", [])):
        d = df.copy()
        d["group_id"] = pd.factorize(d[LOOSE + extra].astype(object).fillna("NA").astype(str).agg("|".join, axis=1))[0]
        (Xtr, ytr, _), _, (Xte, yte, _) = (get_split3(0.2, df=d)[k] for k in ("train", "val", "test"))
        for m in ("random_forest", "xgboost"):
            p = make_pipeline(clone(get_specs()[m][0])).fit(Xtr, ytr)
            rows.append({"grouping": name, "n_groups": int(d.group_id.nunique()), "model": m, **E.metrics(yte, p.predict(Xte))})
            print(rows[-1], flush=True)
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(C.REPORTS / "group_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
