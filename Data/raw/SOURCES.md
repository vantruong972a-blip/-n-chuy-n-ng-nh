# Nguồn dữ liệu thật

| File | Nguồn gốc | Số bản ghi gốc | Ghi chú |
|---|---|---|---|
| `bonbanh_used_cars.json` | tin rao xe cũ trên bonbanh.com | 28.957 | giá VND; không có tỉnh/thành, không có ngày đăng |
| `chotot_used_cars.csv`   | tin rao xe cũ trên xe.chotot.com | 21.927 | giá VND; có tỉnh/thành (suy ra từ URL tin) |

Hai file được thu thập (crawl) bởi dự án mã nguồn mở
<https://github.com/Hyprnx/used-cars-prices-prediction> (commit `d0ee2b5afae0b4bf008e59f46299309f5154cb9a`,
giấy phép GPL-3.0) và **được giữ nguyên, không sửa** trong thư mục này. Mọi bước làm sạch nằm trong
`src/preprocess.py` và ghi ra `Data/processed/`.

Tải lại / kiểm tra toàn vẹn: `python scripts/download_data.py` (so SHA-256 với `CHECKSUMS.sha256`).

## Hạn chế cần biết
- Đây là **giá niêm yết** (người bán rao), thường cao hơn giá chốt; dữ liệu thu thập khoảng 2022.
- Dữ liệu do bên thứ ba crawl từ các website thương mại: hãy kiểm tra điều khoản sử dụng của bonbanh/chotot
  và giấy phép GPL-3.0 của repo gốc trước khi dùng cho mục đích ngoài học tập/nghiên cứu.
- Không có cột ngày đăng, mã lực (horsepower); `variant` chỉ suy ra được cho bonbanh; `province` chỉ có cho chotot.
- Không thể phát hiện cùng một xe đăng trên cả hai website (không có số khung/biển số).
- Các nguồn khác trong repo gốc không dùng: `sanxehot` (không có hãng/số chỗ, năm là chuỗi, nhiều giá trùng
  lặp bất thường), `xetot` (chỉ 390 tin).
