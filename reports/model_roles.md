# Vai trò đo được của từng thuật toán (tập validation, mô hình fit trên train)

Sai số tương đối: RMSLE (log-giá, nhỏ hơn là tốt hơn) và sai số phần trăm trung vị (medAPE). 'Thắng' = thấp nhất trong phân khúc.

| Mô hình | Vai trò | Thắng RMSLE | Thắng medAPE | RMSLE toàn bộ | medAPE toàn bộ (%) |
|---|---|---|---|---|---|
| Linear Regression | Quy luật khấu hao | 0/13 | 0/13 | 0.2362 | 7.59 |
| Random Forest | Đối chứng ổn định | 1/13 | 11/13 | 0.2285 | 5.86 |
| XGBoost | Định giá chính | 12/13 | 2/13 | 0.2134 | 6.44 |

Mô hình mặc định: **XGBoost**. Khuyến nghị theo giá dự đoán (chỉ đổi khi tốt hơn >= 3%):

| Giá dự đoán | n | Khuyến nghị | RMSLE từng mô hình |
|---|---|---|---|
| dưới 300 triệu | 2488 | XGBoost | Linear Regression 0.2728, Random Forest 0.2682, XGBoost 0.2526 |
| 300 triệu – 500 triệu | 2415 | XGBoost | Linear Regression 0.2126, Random Forest 0.201, XGBoost 0.1914 |
| 500 triệu – 800 triệu | 2126 | XGBoost | Linear Regression 0.2036, Random Forest 0.2017, XGBoost 0.1867 |
| 800 triệu – 1.2 tỷ | 1046 | XGBoost | Linear Regression 0.2458, Random Forest 0.2285, XGBoost 0.2304 |
| 1.2 tỷ – 2 tỷ | 593 | XGBoost | Linear Regression 0.199, Random Forest 0.186, XGBoost 0.1624 |
| trên 2 tỷ | 516 | XGBoost | Linear Regression 0.2899, Random Forest 0.2835, XGBoost 0.224 |
