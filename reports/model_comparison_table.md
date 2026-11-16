| Mô hình | Cấu hình dùng | Val RMSE baseline (tr) | Val RMSE tuned (tr) | Val RMSLE | Test MAE (tr) | Test RMSE (tr) | Test RMSLE | Test R² | MAPE % | medAPE % | trong ±20% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Linear Regression | tuned | 744.9 | 510.0 | 0.2362 | 122.4 | 632.8 | 0.2370 | 0.861 | 13.9 | 7.6 | 85% |
| Random Forest | tuned | 547.6 | 521.5 | 0.2285 | 101.9 | 596.9 | 0.2170 | 0.877 | 12.1 | 5.6 | 88% |
| XGBoost ⭐ | tuned | 440.2 | 421.6 | 0.2134 | 103.0 | 558.7 | 0.2077 | 0.892 | 12.2 | 6.5 | 87% |
| (tham chiếu: trung vị giá) | - | - | - | - | 561.9 | 1,738.9 | 0.9282 | -0.047 | 89.3 | 47.7 | 23% |
