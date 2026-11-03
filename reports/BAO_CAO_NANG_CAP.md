# Báo cáo nâng cấp dự án (sinh tự động từ file kết quả)

## 1. Dữ liệu

- Bản ghi thô: **50,884** (bonbanh 28,957, chotot 21,927).
- Sau làm sạch (có nhãn giá): **45,510** (bonbanh 24,926, chotot 20,584); tập chưa nhãn: 253; số nhóm tin rao (group_id): 21,697.
- Trước nâng cấp: dữ liệu MÔ PHỎNG 205 dòng (đã chuyển sang `Data/demo/`, không dùng để huấn luyện).

| Bước làm sạch | Loại | Còn lại |
|---|---:|---:|
| loai_tin_trung_lap | 5,074 | 45,810 |
| gia_thieu_hoac_khong_duong | 1 | 45,809 |
| nam_san_xuat_khong_hop_le | 21 | 45,788 |
| thieu_hang_hoac_dong_xe | 0 | 45,788 |
| xe_khach_>16_cho | 7 | 45,781 |
| hang_dong_xe_khong_khop | 18 | 45,763 |
| gia_ngoai_khoang_10tr_100ty | 149 | 45,614 |
| gia_lech_>15_lan_trung_vi_dong_xe | 104 | 45,510 |

Sửa mức trường (giữ dòng, đặt về thiếu): mileage_km_zero_or_missing_to_nan=12,962, mileage_km_over_1M_to_nan=125, mileage_km_over_100k_per_year_to_nan=204, engine_capacity_implausible_to_nan=0, seats_lt2_to_nan=1

Lý do loại dòng: duplicate_listing=5,074, price_out_of_vnd_range=149, price_ratio_vs_model_median=104, year_invalid=21, brand_model_mismatch=18, out_of_scope_bus=7, price_missing_or_nonpositive=1

Giá (VND) sau làm sạch: trung vị 480 triệu; p1 47 triệu; p99 8.05 tỷ; max 68.0 tỷ.

## 2. Kết quả mô hình (tập test giữ riêng, chia theo nhóm)

| Mô hình | Cấu hình dùng | Val RMSE baseline (tr) | Val RMSE tuned (tr) | Test MAE (tr) | Test RMSE (tr) | Test R² | MAPE % | medAPE % | trong ±20% |
|---|---|---|---|---|---|---|---|---|---|
| Linear Regression | tuned | 943.5 | 666.3 | 142.3 | 700.4 | 0.830 | 15.3 | 8.3 | 80% |
| Random Forest | baseline | 537.4 | 585.5 | 89.4 | 481.8 | 0.920 | 11.9 | 5.2 | 88% |
| XGBoost ⭐ | tuned | 440.2 | 435.9 | 98.8 | 481.3 | 0.920 | 12.2 | 6.4 | 88% |
| (tham chiếu: trung vị giá) | - | - | - | 561.9 | 1,738.9 | -0.047 | 89.3 | 47.7 | 23% |


Tất cả tỷ lệ chia đã chạy:

| Tỷ lệ | Mô hình | Cấu hình | Val RMSE (tr) | Test MAE (tr) | Test RMSE (tr) | R² | medAPE % | Chọn |
|---|---|---|---:|---:|---:|---:|---:|:-:|
| 60/40 | linear_regression | tuned | 551.7 | 142.8 | 697.5 | 0.827 | 8.5 |  |
| 60/40 | random_forest | baseline | 372.2 | 94.4 | 550.4 | 0.892 | 5.5 | ✔ |
| 60/40 | xgboost | tuned | 374.1 | 114.3 | 566.5 | 0.886 | 7.8 |  |
| 70/30 | linear_regression | tuned | 619.9 | 134.2 | 663.4 | 0.835 | 8.2 |  |
| 70/30 | random_forest | baseline | 395.4 | 89.9 | 579.5 | 0.874 | 5.3 | ✔ |
| 70/30 | xgboost | tuned | 401.9 | 96.0 | 499.5 | 0.907 | 6.3 |  |
| 80/20 | linear_regression | tuned | 666.3 | 142.3 | 700.4 | 0.830 | 8.3 |  |
| 80/20 | random_forest | baseline | 537.4 | 89.4 | 481.8 | 0.920 | 5.2 |  |
| 80/20 | xgboost | tuned | 435.9 | 98.8 | 481.3 | 0.920 | 6.4 | ✔ |
| 90/10 | linear_regression | tuned | 519.4 | 134.3 | 625.3 | 0.872 | 8.2 |  |
| 90/10 | random_forest | tuned | 502.4 | 99.0 | 677.9 | 0.850 | 4.9 |  |
| 90/10 | xgboost | baseline | 440.0 | 120.8 | 709.4 | 0.836 | 7.8 | ✔ |

Mô hình phục vụ API: **xgboost** (phiên bản `20261025-041248-d5e70f1`, dữ liệu `5e70f1b5fcc3`).

### So sánh trước/sau (CHỈ mang tính tham khảo)

Kết quả cũ trên dữ liệu mô phỏng (USD, test 41 dòng) KHÔNG so sánh trực tiếp được với kết quả mới (VND, dữ liệu thật). Chỉ R² là không phụ thuộc đơn vị, nhưng vẫn là hai bài toán khác nhau.

| Mô hình | R² cũ (mô phỏng, 80/20) | R² mới (thật, 80/20) |
|---|---:|---:|
| linear_regression | 0.907 | 0.830 |
| random_forest | 0.907 | 0.920 |
| xgboost | 0.924 | 0.920 |

### Đối chứng biến mục tiêu (validation, tham số mặc định)

| Mô hình | Mục tiêu | MAE (tr) | RMSE (tr) | R² | medAPE % |
|---|---|---:|---:|---:|---:|
| linear_regression | log1p(price) | 169.3 | 943.5 | 0.665 | 9.2 |
| linear_regression | price (VND) | 332.4 | 987.4 | 0.633 | 31.7 |
| random_forest | log1p(price) | 95.2 | 537.4 | 0.891 | 5.7 |
| random_forest | price (VND) | 88.7 | 486.8 | 0.911 | 5.5 |
| xgboost | log1p(price) | 106.1 | 440.2 | 0.927 | 7.4 |
| xgboost | price (VND) | 103.2 | 576.7 | 0.875 | 8.6 |

## 3. Permutation importance (xgboost, test)

Xáo trộn một cột (hoặc cả nhóm cột tương quan, dòng [nhóm]) rồi đo RMSE tăng thêm. Các cột tương quan (hãng/dòng/phiên bản) chia sẻ vai trò nên xem cả dòng nhóm; đây là mức phụ thuộc của mô hình, không phải quan hệ nhân quả.

| Đặc trưng | RMSE tăng khi xáo (triệu VND) |
|---|---:|
| [nhóm] thông số máy (engine_capacity+transmission+fuel_type+drivetrain+seats) | 1,251.1 |
| engine_capacity | 1,050.3 |
| [nhóm] mức sử dụng (year+mileage_km) | 910.2 |
| year | 734.1 |
| [nhóm] định danh xe (brand+model+variant+body_type) | 674.3 |
| brand | 566.9 |
| transmission | 494.3 |
| [nhóm] thị trường (origin+province) | 301.0 |
| origin | 278.8 |
| drivetrain | 216.3 |
| mileage_km | 156.0 |
| model | 95.5 |

## 4. Hiệu năng theo nhóm (mô hình tốt nhất, test)

| Nhóm theo | Giá trị | n | MAE (tr) | RMSE (tr) | R² | medAPE % |
|---|---|---:|---:|---:|---:|---:|
| age_band | 0-2 năm | 1899 | 160.7 | 682.1 | 0.947 | 4.8 |
| age_band | 3-5 | 2591 | 86.8 | 282.2 | 0.929 | 5.3 |
| age_band | 6-10 | 2398 | 105.7 | 561.4 | 0.838 | 6.7 |
| age_band | 11-20 | 2206 | 58.1 | 373.7 | 0.514 | 9.3 |
| age_band | >20 | 231 | 42.2 | 163.8 | 0.177 | 20.9 |
| price_band | <300tr | 2522 | 26.9 | 53.6 | 0.474 | 9.1 |
| price_band | 300-600tr | 3292 | 35.8 | 78.4 | 0.156 | 5.2 |
| price_band | 0.6-1 tỷ | 1824 | 61.9 | 99.8 | 0.194 | 5.3 |
| price_band | 1-2 tỷ | 1085 | 145.2 | 246.5 | 0.174 | 6.8 |
| price_band | >2 tỷ | 602 | 772.5 | 1,844.6 | 0.851 | 9.6 |
| mileage_known | có km | 6657 | 95.0 | 474.6 | 0.859 | 6.8 |
| mileage_known | không có km | 2668 | 108.2 | 497.7 | 0.959 | 5.6 |
| source | bonbanh | 4892 | 112.8 | 618.6 | 0.923 | 5.5 |
| source | chotot | 4433 | 83.3 | 254.9 | 0.869 | 7.6 |

## 5. 10 ca sai tuyệt đối lớn nhất (test)

| Nguồn | Hãng | Dòng | Năm | Km | Giá thực (tr) | Dự đoán (tr) |
|---|---|---|---:|---:|---:|---:|
| bonbanh | rolls-royce | cullinan | 2020 | 8,000 | 40,500.0 | 23,085.0 |
| bonbanh | rolls-royce | phantom | 2014 | 19,000 | 31,800.0 | 16,062.8 |
| bonbanh | mercedes-benz | maybach | 2022 | - | 28,000.0 | 15,411.9 |
| bonbanh | porsche | panamera | 2010 | - | 13,680.0 | 1,390.2 |
| bonbanh | rolls-royce | cullinan | 2020 | 9,999 | 18,799.0 | 31,029.2 |
| bonbanh | rolls-royce | phantom | 2015 | 15,000 | 28,000.0 | 16,841.0 |
| bonbanh | rolls-royce | phantom | 2015 | 13,000 | 28,000.0 | 16,909.6 |
| bonbanh | rolls-royce | phantom | 2008 | 39,000 | 15,688.0 | 4,639.2 |
| bonbanh | land rover | range rover | 2019 | - | 20,000.0 | 10,373.8 |
| chotot | audi | q7 | 2012 | 95,000 | 9,500.0 | 858.6 |

## 6. Hạn chế

- Giá là **giá niêm yết** (người bán rao) thu thập khoảng 2022, không phải giá giao dịch; không phản ánh giá hiện tại.
- Không có ngày đăng, mã lực; `variant` chỉ có ở bonbanh, `province` chỉ có ở chotot (giá trị thiếu được mô hình xử lý như một nhóm).
- bonbanh ghi km = 0 cho 12,962 tin, được coi là 'chưa khai báo' (thiếu), không phải xe chạy 0 km.
- Tên dòng xe không hoàn toàn thống nhất giữa 2 nguồn (vd chotot chỉ giữ từ đầu: `cx` so với `cx5`).
- Không phát hiện được cùng một xe đăng trên cả hai website; tin đăng lặp trong cùng nguồn được gom nhóm khi chia dữ liệu.
- Xe hiếm/siêu sang có rất ít mẫu nên sai số lớn (vd Rolls-Royce, Maybach); ba mô hình có thể lệch nhau hàng chục tỷ VND cho các xe này. R² trong từng phân khúc giá thấp một phần vì phương sai giá trong mỗi dải hẹp, hãy đọc cùng medAPE.
- Một số giá nghi sai vẫn còn vì không đủ bằng chứng để loại theo quy tắc (vd Audi Q7 2012 giá 9,5 tỷ trên chotot, chỉ lệch ~10 lần trung vị dòng xe < ngưỡng 15 lần); chúng làm tăng sai số test.
- Mô hình tốt nhất đổi giữa các tỷ lệ chia (chọn theo validation); chênh lệch RF/XGBoost ở 80/20 rất nhỏ (xem bảng), không nên coi một mô hình là vượt trội chắc chắn.
- Dữ liệu do bên thứ ba crawl; kiểm tra điều khoản sử dụng và giấy phép GPL-3.0 của repo gốc khi dùng ngoài mục đích học tập.
