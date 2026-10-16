# MLOps: Dự đoán giá xe

Đồ án cơ sở MLOps: dự đoán giá xe (`price`) từ đặc trưng kỹ thuật, kết hợp học có giám sát, không giám sát, bán giám sát, theo dõi thí nghiệm (MLflow), API + web (FastAPI), Docker, CI và giám sát cơ bản.

> ⚠️ **Dữ liệu hiện tại là dữ liệu MÔ PHỎNG** (`src/make_demo_data.py`, xem `Data/raw/DEMO_DATA_NOTICE.txt`) để chạy thử pipeline. Mọi chỉ số trong repo này **không có giá trị khoa học**. Hãy thay bằng dữ liệu xe thật (xem mục 4) rồi chạy lại toàn bộ pipeline trước khi viết báo cáo.

## 1. Mục tiêu và chức năng
- Tiền xử lý tái sử dụng cho huấn luyện và dự đoán (`ColumnTransformer` + `Pipeline`, chỉ fit trên train).
- So sánh Linear Regression, Random Forest, XGBoost trên 4 tỷ lệ train/test (60/40, 70/30, 80/20, 90/10), có tinh chỉnh siêu tham số bằng cross-validation chỉ trên tập train.
- Học không giám sát (PCA, K-Means, DBSCAN) và bán giám sát (Self-Training) có kiểm soát.
- Theo dõi thí nghiệm bằng MLflow; API FastAPI + giao diện web; Docker; pytest; GitHub Actions; log, độ trễ, lỗi, data drift, sai số theo nhãn thực tế.

## 2. Kiến trúc và vòng đời MLOps
```
Data/raw/car.csv ──► preprocess.py (làm sạch, mã dữ liệu) ──► train_supervised.py
                                   │                          │  Pipeline(tiền xử lý + mô hình), CV trên train
                                   │                          ├─► MLflow (mlruns/): tham số, chỉ số, mô hình
                                   │                          ├─► reports/*.csv, *.png
                                   │                          └─► models/*.joblib + model_info.json + reference_stats.json
unsupervised.py / semi_supervised.py ──► reports/
models/ ──► inference.py ──► app/main.py (FastAPI) ──► web  ──► logs/*.jsonl ──► /api/monitoring (drift, độ trễ, lỗi, MAE thực tế)
                                         ▲ GitHub Actions: ruff + pytest        ▲ Docker
```
Vòng đời: dữ liệu → huấn luyện/đánh giá → lưu phiên bản (`model_version` = thời gian + mã dữ liệu) → triển khai API → giám sát → huấn luyện lại khi cần.

## 3. Cấu trúc thư mục
| Đường dẫn | Chức năng |
|---|---|
| `Data/raw/` | Dữ liệu gốc `car.csv` (không sửa trực tiếp) |
| `Data/external/` | `car_unlabeled.csv` (không có `price`, tùy chọn cho bán giám sát) |
| `Data/processed/` | `clean.csv` (đã làm sạch, chưa scale), `data_info.json` (số dòng, cột, mã phiên bản dữ liệu) |
| `Data/splits/` | `split_summary.csv`, chỉ số train/test của tỷ lệ 80/20 |
| `Data/images/` | Chỉ dùng khi có ảnh xe thật (hiện trống) |
| `src/config.py` | Đường dẫn, `RANDOM_STATE`, tỷ lệ chia, đơn vị giá |
| `src/preprocess.py` | Làm sạch, `build_preprocessor`, `get_split`, mã phiên bản dữ liệu |
| `src/model_registry.py` | Danh sách mô hình + lưới siêu tham số (thêm mô hình mới ở đây) |
| `src/train_supervised.py` | Huấn luyện/so sánh, lưu mô hình, biểu đồ, MLflow |
| `src/unsupervised.py` | PCA, K-Means, DBSCAN |
| `src/semi_supervised.py` | Self-Training |
| `src/tracking.py` | Bọc MLflow (không có mlflow thì bỏ qua và cảnh báo) |
| `src/inference.py` | Nạp mô hình, kiểm tra đầu vào, dự đoán |
| `src/monitoring.py` | Log, drift (PSI), sai số theo nhãn thực tế |
| `src/make_demo_data.py` | Tạo dữ liệu mô phỏng (chỉ để chạy thử) |
| `app/main.py`, `app/static/index.html` | FastAPI + giao diện web |
| `models/` | Mô hình đã lưu (Pipeline), `model_info.json`, `reference_stats.json` |
| `reports/` | Kết quả CSV/JSON và biểu đồ |
| `tests/` | pytest (tiền xử lý, huấn luyện, suy luận, monitoring, API) |
| `scripts/` | Script PowerShell/bash chạy pipeline, API, MLflow |
| `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` | Đóng gói và CI |

## 4. Nguồn dữ liệu và ý nghĩa cột
Dữ liệu hiện tại: **mô phỏng**, 205 dòng có nhãn + 150 dòng không nhãn. Để dùng dữ liệu thật (ví dụ bộ *Car Price Prediction* trên Kaggle hoặc dữ liệu của trường), đặt file vào `Data/raw/car.csv` với cột mục tiêu `price` (tên cột không phân biệt hoa thường), rồi **ghi rõ nguồn dữ liệu và đơn vị giá tại đây**. Nếu tên cột khác, hãy đổi tên cột hoặc sửa `DROP_COLS`/`TARGET` trong `src/config.py`. Đặt đơn vị đúng bằng biến môi trường, ví dụ `$env:MLOPS_PRICE_UNIT="INR"` (hoặc sửa `PRICE_UNIT` trong `config.py`).

| Cột | Ý nghĩa |
|---|---|
| `car_id`, `carname` | Định danh, **bị loại** khỏi đặc trưng |
| `fueltype`, `aspiration`, `carbody`, `drivewheel` | Đặc trưng phân loại (one-hot) |
| `enginesize`, `horsepower`, `curbweight`, `citympg`, `highwaympg` | Đặc trưng số (chuẩn hóa) |
| `price` | Cột mục tiêu |

Xử lý: bỏ dòng trùng (sau khi bỏ cột định danh), bỏ dòng thiếu/≤0 giá, điền thiếu (median / giá trị phổ biến nhất) trong Pipeline, one-hot với `handle_unknown="ignore"`.

## 5. Cài đặt (Windows PowerShell)
```powershell
cd MLOPS
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # nếu bị chặn: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements-dev.txt
```

## 6. Chạy pipeline
Chạy tất cả: `.\scripts\run_pipeline.ps1` (thêm `-Fast` để chạy thử nhanh). Hoặc từng bước:
```powershell
python src\preprocess.py                         # làm sạch + thông tin dữ liệu
python src\train_supervised.py --all             # 4 tỷ lệ, 3 mô hình, có tinh chỉnh (CV)
python src\unsupervised.py
python src\semi_supervised.py
```
Tỷ lệ train/test: `--all` (60/40, 70/30, 80/20, 90/10) hoặc `--test-size 0.3` (70/30). Tùy chọn khác: `--models random_forest xgboost`, `--no-tune`, `--fast`, `--no-mlflow`.
Mô hình dùng cho API được lưu từ tỷ lệ **80/20** (`SERVE_TEST_SIZE`). Mô hình "tốt nhất" chọn theo **CV-RMSE thấp nhất trên tập train**; chỉ số test chỉ để báo cáo cuối.

**Tính đúng đắn:** bộ tiền xử lý nằm trong `Pipeline` cùng mô hình nên chỉ fit trên train; API nạp đúng Pipeline đã lưu và sắp cột theo `model_info.json`. Lưu ý: tỷ lệ **90/10** với dataset nhỏ cho tập test rất ít dòng nên chỉ số dao động mạnh, không mặc định tốt hơn tỷ lệ khác.

## 7. Học không giám sát (`src/unsupervised.py`)
- **PCA**: giảm số chiều xuống 2 để trực quan hóa, cho biết phương sai giữ lại.
- **K-Means**: chia xe thành k nhóm có đặc điểm gần nhau; k chọn theo silhouette cao nhất (kèm elbow/Davies-Bouldin ở `k_selection.png`).
- **DBSCAN**: tìm nhóm theo mật độ và điểm nhiễu (nhãn -1); `eps` chọn bằng lưới phân vị khoảng cách láng giềng, ràng buộc ≥2 cụm và nhiễu ≤20%.
- Chỉ fit trên **train**; test chỉ được chiếu/gán cụm. **Không dùng `price`** để phân cụm; `price` chỉ dùng sau đó để mô tả cụm (`cluster_price_summary.csv`). Phân cụm **không phải** dự đoán giá.
- Kết quả: `reports/unsupervised_labels.csv`, `unsupervised_metrics.json`, `unsupervised.png`.

## 8. Học bán giám sát (`src/semi_supervised.py`)
Ẩn có kiểm soát 30/50/70/90% nhãn của **tập train** (nhãn test không bị đụng), nhiều seed. So sánh: baseline (chỉ nhãn có), self-training (Random Forest, nhận pseudo-label có độ bất định thấp nhất theo độ lệch chuẩn giữa các cây, trọng số 0.5) và oracle (đủ nhãn). Tùy chọn `--use-external` thêm `car_unlabeled.csv` vào pool không nhãn. Kết quả: `reports/semi_supervised_results.csv`, `semi_supervised_summary.csv`, `semi_supervised.png`.
**Hạn chế:** pseudo-label có thể khuếch đại sai số của mô hình ban đầu. Trong lần chạy trên dữ liệu mô phỏng, self-training chỉ nhỉnh hơn baseline ở mức ẩn 30% và kém hơn hoặc dao động ở các mức khác; **không khẳng định bán giám sát luôn tốt hơn**.

## 9. MLflow
```powershell
python src\train_supervised.py --all
mlflow ui --backend-store-uri ./mlruns --port 5000     # hoặc .\scripts\start_mlflow.ps1
```
Mở http://127.0.0.1:5000, experiment `car-price-prediction`: mỗi (mô hình × tỷ lệ chia) là một run, ghi tên thuật toán, tỷ lệ chia, siêu tham số, MAE/RMSE/R², CV-RMSE, thời gian huấn luyện, số dòng train/test, `data_version`, `model_version` và artifact mô hình (Pipeline). Không dùng Model Registry (chưa cần cho đồ án cơ sở).

## 10. API và web
```powershell
uvicorn app.main:app --reload       # hoặc .\scripts\start_api.ps1
```
Web: http://127.0.0.1:8000 · Tài liệu API: http://127.0.0.1:8000/docs

| Endpoint | Chức năng |
|---|---|
| `GET /health` | Trạng thái; 503 nếu chưa có mô hình |
| `GET /api/meta` | Danh sách đặc trưng, mô hình, đơn vị giá, phiên bản |
| `GET /api/metrics` · `/api/metrics/all` | Chỉ số mô hình đang phục vụ · toàn bộ thí nghiệm |
| `POST /api/predict` | Body `{"values": {...}, "model": "random_forest"}`; trả giá, đơn vị, tên + phiên bản mô hình, giá của mọi mô hình trên **cùng một đầu vào** |
| `POST /api/feedback` | Gửi giá thực tế `{"prediction_id","actual_price"}` |
| `GET /api/monitoring` | Số yêu cầu, độ trễ, lỗi, drift, sai số theo nhãn thực tế |

Đầu vào thiếu, sai kiểu, ngoài phạm vi hợp lý (min/max train ± 50% khoảng giá trị), giá trị phân loại lạ, trường thừa hoặc tên mô hình sai đều trả **422** kèm danh sách lỗi.

## 11. Giám sát
Log nằm ở `logs/` (`predictions.jsonl`, `errors.jsonl`, `feedback.jsonl`). **Data drift** = phân phối đầu vào lệch so với train, đo bằng PSI (>0,2 là dấu hiệu, cần ≥30 yêu cầu). **Concept drift** (quan hệ đầu vào–giá đổi) chỉ đánh giá được khi có nhãn thực tế qua `/api/feedback`; data drift **không** đồng nghĩa concept drift. Khi chưa có nhãn mới, API báo rõ rằng chưa tính được sai số thực tế.

## 12. Docker và kiểm thử
```powershell
python src\train_supervised.py --test-size 0.2   # cần có models/ trước khi build
docker build -t car-price-api .
docker run -p 8000:8000 car-price-api
docker compose up --build                          # API (8000) + MLflow UI (5000)
pytest -q                                          # kiểm thử
ruff check src app tests --select E9,F --ignore F401
```
CI (`.github/workflows/ci.yml`) chạy ruff, kiểm tra import và pytest khi push/pull request; không triển khai cloud, không cần bí mật.

## 13. Kết quả thực nghiệm và hạn chế
Các file kết quả nằm ở `reports/` (sinh ra từ lần chạy thực tế, không hardcode). Lần chạy kèm theo gói này dùng **dữ liệu mô phỏng** và **chưa có XGBoost/MLflow** (môi trường tạo gói không cài được), nên chỉ có Linear Regression và Random Forest. Bạn cần chạy lại trên máy mình để có XGBoost và MLflow.

Ví dụ (dữ liệu mô phỏng, RMSE trên test):

| Tỷ lệ | Linear Regression | Random Forest | Chọn theo CV-RMSE |
|---|---|---|---|
| 60/40 | 1638 | 1563 | Linear Regression |
| 70/30 | 1601 | 1459 | Linear Regression |
| 80/20 | 1535 | 1538 | Random Forest |
| 90/10 (test chỉ 21 dòng) | 1769 | 1919 | Linear Regression |

Hạn chế: dữ liệu mô phỏng (giá sinh gần như tuyến tính); dataset nhỏ nên chỉ số dao động; đơn vị giá chỉ là giả định; chưa có ảnh xe; Model Registry chưa triển khai; phần API/Docker/CI chưa được chạy kiểm chứng trong môi trường tạo gói (chỉ phần tiền xử lý, huấn luyện, suy luận, monitoring có test chạy qua).

## 14. Lỗi thường gặp
| Lỗi | Cách xử lý |
|---|---|
| `ModuleNotFoundError: xgboost/mlflow/fastapi` | `pip install -r requirements-dev.txt` |
| API trả 503 "Thiếu models/..." | Chạy `python src\train_supervised.py --test-size 0.2` |
| `Thiếu cột mục tiêu 'price'` | Đổi tên cột giá thành `price` |
| Cảnh báo phiên bản scikit-learn khi nạp `.joblib` | Huấn luyện lại bằng phiên bản đang dùng |
| `Activate.ps1` bị chặn | `Set-ExecutionPolicy -Scope Process Bypass` |
| MLflow UI trống | Chạy từ thư mục gốc và dùng `--backend-store-uri ./mlruns` |
| Cổng 8000/5000 bị chiếm | Đổi `--port` |
