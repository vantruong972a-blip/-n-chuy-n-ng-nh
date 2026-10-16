"""Tạo dữ liệu MÔ PHỎNG để chạy thử pipeline khi chưa có car.csv thật.

KHÔNG dùng kết quả từ dữ liệu này cho báo cáo. Thay Data/raw/car.csv bằng
file thật rồi chạy lại các bước.
"""
import numpy as np
import pandas as pd

import config as C


def make(n, rng, with_price=True):
    engine = rng.normal(130, 40, n).clip(60, 330)
    hp = (engine * 0.85 + rng.normal(0, 14, n)).clip(48, 290)
    weight = (1100 + engine * 7.5 + rng.normal(0, 150, n)).clip(1500, 4100)
    city = (46 - weight / 110 + rng.normal(0, 2, n)).clip(12, 50)
    highway = (city + 6 + rng.normal(0, 2, n)).clip(16, 56)
    df = pd.DataFrame({
        "car_id": np.arange(1, n + 1),
        "carname": [f"brand{rng.randint(1, 20)} model{rng.randint(1, 50)}" for _ in range(n)],
        "fueltype": rng.choice(["gas", "diesel"], n, p=[0.9, 0.1]),
        "aspiration": rng.choice(["std", "turbo"], n, p=[0.8, 0.2]),
        "carbody": rng.choice(["sedan", "hatchback", "wagon", "hardtop", "convertible"], n,
                              p=[0.45, 0.3, 0.15, 0.05, 0.05]),
        "drivewheel": rng.choice(["fwd", "rwd", "4wd"], n, p=[0.6, 0.35, 0.05]),
        "enginesize": engine.round(0), "horsepower": hp.round(0),
        "curbweight": weight.round(0), "citympg": city.round(0), "highwaympg": highway.round(0),
    })
    if with_price:
        price = (-9000 + 62 * engine + 55 * hp + 2.6 * weight - 80 * city
                 + 2500 * (df.aspiration == "turbo") + 3000 * (df.drivewheel == "rwd")
                 + rng.normal(0, 1500, n)).clip(5000, 46000)
        df["price"] = price.round(0)
    return df


def main():
    rng = np.random.RandomState(C.RANDOM_STATE)
    (C.DATA / "raw").mkdir(parents=True, exist_ok=True)
    (C.DATA / "external").mkdir(parents=True, exist_ok=True)
    make(205, rng).to_csv(C.RAW_CSV, index=False)
    make(150, rng, with_price=False).to_csv(C.UNLABELED_CSV, index=False)
    (C.DATA / "raw" / "DEMO_DATA_NOTICE.txt").write_text(
        "car.csv va car_unlabeled.csv la du lieu MO PHONG. Thay bang du lieu that.\n")
    print("Da tao du lieu mo phong.")


if __name__ == "__main__":
    main()
