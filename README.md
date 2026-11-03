# MLOps: Dự đoán giá xe ô tô cũ tại Việt Nam

Đồ án MLOps: dự đoán **giá niêm yết** xe ô tô cũ (VND) từ dữ liệu tin rao thật, kết hợp học có giám sát (Linear Regression, Random Forest, XGBoost), không giám sát (PCA, K-Means, DBSCAN), bán giám sát (Self-Training), theo dõi thí nghiệm (MLflow), API + web (FastAPI), Docker, CI (GitHub Actions) và giám sát cơ bản.

> **Lưu ý trung thực.** Mọi con số kết quả nằm trong `reports/` và được sinh ra từ lần chạy thật (xem `reports/BAO_CAO_NANG_CAP.md`); README không chép số cố định. Giá là **giá rao**, không phải giá giao dịch, thu thập khoảng **năm 2022**. Dữ liệu mô phỏng cũ (205 dòng) đã được chuyển sang `Data/demo/` và **không còn dùng để huấn luyện**.

## 1. Những gì đã nâng cấp
| Hạng mục | Trước | Sau |
|---|---|---|
| Dữ liệu | mô phỏng, 205 dòng, USD | thật (bonbanh + chotot), hàng chục nghìn tin, VND |
| Đặc trưng | 9 cột kỹ thuật (UCI) | hãng, dòng, phiên bản, kiểu dáng, năm, km, dung tích, hộp số, nhiên liệu, dẫn động, số chỗ, xuất xứ, tỉnh/thành + `car_age`, `km_per_year`, `km_reported`, `condition`, `brand_model` |
| Chia dữ liệu | ngẫu nhiên train/test | train/validation/test **theo nhóm tin rao** (tin đăng lại cùng xe luôn cùng một tập) |
| Huấn luyện | GridSearch trên train | baseline vs tuned (RandomizedSearch + GroupKFold trên train), chọn theo **validation**, đo test một lần |
| Đánh giá | MAE/RMSE/R² | + MAPE, medAPE, % trong ±20%, permutation importance, hiệu năng theo nhóm xe, 30 ca sai nhiều nhất, đối chứng biến mục tiêu |
| MLflow | hàm `log` bị cắt (chỉ log mô hình) | log tham số, chỉ số, artifact, mô hình, `data_version` |
| Triển khai mô hình | ghi đè | huấn luyện → kiểm tra cổng → lưu bản cũ vào `models/archive/` → mới thay |

## 2. Kiến trúc
```
Data/raw/*.json,*.csv ─► data_sources.py (chuẩn hóa lược đồ) ─► preprocess.py (làm sạch, loại dòng có lý do, group_id, chia nhóm)
        ▲ scripts/download_data.py (SHA-256)                      │ Data/processed/{clean,unlabeled,rejected_rows}.csv, cleaning_report.json
                                                                   ▼
features.py (FeatureEngineer + ColumnTransformer, nằm TRONG Pipeline) ─► train_supervised.py ─► MLflow (mlflow.db, mlruns/)
                                                                   │      └► reports/*.csv, *.png, *.md
                                                                   └► models/versions/<phiên bản>/ ─(qua cổng kiểm tra)─► models/*.joblib + model_info.json + reference_stats.json
unsupervised.py / semi_supervised.py ─► reports/        models/ ─► inference.py ─► app/main.py (FastAPI) ─► web ─► logs/*.jsonl ─► /api/monitoring
```

## 3. Cấu trúc thư mục
| Đường dẫn | Chức năng |
|---|---|
| `Data/raw/` | Dữ liệu gốc **không sửa**: `bonbanh_used_cars.json`, `chotot_used_cars.csv`, `CHECKSUMS.sha256`, `SOURCES.md` (nguồn, giấy phép, hạn chế) |
| `Data/processed/` | `clean.csv` (có nhãn, đã làm sạch), `unlabeled.csv` (tin thật có giá bị loại vì không đáng tin), `rejected_rows.csv` (mỗi dòng bị loại kèm lý do), `cleaning_report.json`, `data_info.json` |
| `Data/splits/` | `split_summary.csv` và chỉ số `train/val/test_idx.csv` của tỷ lệ 80/20 |
| `Data/demo/` | Dữ liệu mô phỏng cũ + kết quả cũ (chỉ để đối chiếu lịch sử) |
| `src/config.py` | Đường dẫn, hằng số, ngưỡng làm sạch, danh sách đặc trưng |
| `src/data_sources.py` | Đọc bonbanh/chotot → lược đồ chuẩn; chuẩn hóa tên hãng/dòng, tỉnh/thành từ URL |
| `src/preprocess.py` | Làm sạch, loại trùng, kiểm tra giá, `group_id`, chia train/val/test theo nhóm, `quality_checks` |
| `src/features.py` | `FeatureEngineer` (đặc trưng phái sinh) + bộ tiền xử lý (impute/scale/one-hot) |
| `src/model_registry.py` | Mô hình, baseline, lưới siêu tham số; bọc `log1p` cho biến mục tiêu |
| `src/train_supervised.py` | Huấn luyện/so sánh/phân tích, lưu phiên bản, cổng kiểm tra, MLflow |
| `src/evaluation.py` | Chỉ số, permutation importance, hiệu năng theo nhóm, ca sai lớn |
| `src/tracking.py` | Bọc MLflow (không có mlflow thì bỏ qua và cảnh báo) |
| `src/unsupervised.py`, `src/semi_supervised.py` | PCA/K-Means/DBSCAN; Self-Training |
| `src/inference.py`, `src/monitoring.py` | Kiểm tra đầu vào + dự đoán; log, drift (PSI), sai số theo nhãn thực tế |
| `app/main.py` | FastAPI: API + phục vụ giao diện React đã build (`frontend/dist`) |
| `app/static/index.html` | Giao diện HTML cũ, vẫn truy cập được ở `/cu` |
| `frontend/` | Giao diện React + TypeScript + Tailwind + Recharts (xem mục 10.1) |
| `scripts/` | `download_data.py`, `run_pipeline.ps1/.sh`, `make_report.py`, `group_sensitivity.py`, `start_api.ps1`, `start_mlflow.ps1` |
| `tests/` | pytest: chất lượng dữ liệu, tiền xử lý, huấn luyện, lưu/nạp mô hình, API, MLflow, monitoring |
| `archive/`, `models/archive/`, `reports/archive_demo_simulated/` | Bản cũ (MLflow/mô hình/kết quả trên dữ liệu mô phỏng) được giữ lại, không xóa |

## 4. Dữ liệu
Nguồn (chi tiết ở `Data/raw/SOURCES.md`): tin rao xe cũ trên **bonbanh.com** (28.957 tin) và **xe.chotot.com** (21.927 tin), do dự án mã nguồn mở `Hyprnx/used-cars-prices-prediction` (GPL-3.0, commit cố định) thu thập; file được chép nguyên vẹn và đối chiếu bằng SHA-256. Hãy kiểm tra điều khoản sử dụng của các website và giấy phép trước khi dùng ngoài mục đích học tập.

| Cột chuẩn | Ý nghĩa / ghi chú |
|---|---|
| `brand`, `model`, `variant` | Hãng, dòng xe, phiên bản (`variant` chỉ suy ra được cho bonbanh từ tiêu đề tin) |
| `body_type` | sedan, suv_crossover, hatchback, pickup, van, mpv, coupe, convertible, other |
| `year` | Năm sản xuất (cột `date` của chotot thực chất là năm sản xuất) |
| `mileage_km` | Số km. **Giá trị 0 / >1.000.000 / >100.000 km mỗi năm được đặt là thiếu** (bonbanh dùng 0 cho "chưa khai báo") |
| `engine_capacity` (lít), `transmission`, `fuel_type`, `drivetrain`, `seats`, `origin` | Thông số xe; `drivetrain` chỉ có ở bonbanh |
| `province` | Tỉnh/thành, suy ra từ URL chotot (bonbanh không có) |
| `price` | **Giá niêm yết, VND** (mục tiêu) |
| `source`, `source_id`, `group_id` | Truy vết; **không** đưa vào mô hình. `group_id` gom các tin đăng lại cùng một xe |
| Không có | Ngày đăng, mã lực (nguồn không cung cấp, nên không bịa) |

**Đặc trưng phái sinh** (trong `features.py`, chỉ tính từ đầu vào, không dùng giá): `car_age = 2022 − year` (2022 là năm thu thập dữ liệu), `km_per_year`, `km_reported`, `condition` (new_like/used/km_unknown), `brand_model`.

**Làm sạch** (`python src/preprocess.py`; số liệu thật ở `Data/processed/cleaning_report.json`): bỏ tin trùng (cùng `source_id` hoặc trùng mọi thuộc tính kể cả giá); bỏ năm ngoài [1980, 2022]; bỏ xe khách >16 chỗ; bỏ cặp hãng–dòng sai rõ ràng (vd `ecosport` gắn `mercedes-benz`); bỏ giá <10 triệu hoặc >100 tỷ VND; bỏ giá lệch >15 lần trung vị cùng dòng xe (nhóm ≥8 tin) — các dòng bị loại vì giá toàn là giá giả (`12345678`, `5 VND`, `420 tỷ` cho Kia Soluto…) và được kiểm tra thủ công. **Không xóa** xe sang/xe hiếm giá cao nhưng nhất quán với dòng xe (vd Rolls-Royce 68 tỷ vẫn được giữ). Tin có đặc trưng hợp lệ nhưng giá bị loại được chuyển sang `unlabeled.csv` (dữ liệu chưa nhãn thật cho bán giám sát).

## 5. Cài đặt trên máy mới (Windows PowerShell)
```powershell
cd MLOPS
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # nếu bị chặn: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements-dev.txt
python scripts\download_data.py       # nếu Data\raw thiếu/hỏng; đối chiếu SHA-256
```
Linux/macOS: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt`.

## 6. Chạy pipeline
Tất cả: `.\scripts\run_pipeline.ps1` (thêm `-Fast` để chạy thử trên mẫu nhỏ — **không** thay mô hình đang phục vụ; `-AllSplits` để chạy 4 tỷ lệ). Hoặc từng bước:
```powershell
python src\preprocess.py                        # làm sạch + chia nhóm + báo cáo
python src\train_supervised.py                  # tỷ lệ phục vụ API (80/20): baseline + tuned, đầy đủ phân tích
python src\train_supervised.py --all            # thêm 60/40, 70/30, 90/10 (rất lâu)
python src\unsupervised.py
python src\semi_supervised.py                   # thêm --use-external để dùng unlabeled.csv
python scripts\make_report.py                   # tổng hợp reports\BAO_CAO_NANG_CAP.md từ file kết quả
```
Tùy chọn của `train_supervised.py`: `--models random_forest xgboost`, `--no-tune`, `--fast`, `--max-rows N` (mẫu con theo nhóm, chỉ để thử), `--n-iter`, `--no-promote`, `--no-mlflow`.

**Quy trình đánh giá:** chia theo `group_id` thành train/validation/test (mặc định 60/20/20 khi test=20%, `RANDOM_STATE=42`). Mỗi mô hình có *baseline* (tham số mặc định) và *tuned* (RandomizedSearchCV + GroupKFold, chỉ trên train); cấu hình dùng được chọn theo RMSE trên **validation**, sau đó fit lại trên train+validation và đo trên **test** — cùng một test cho cả ba mô hình. Mô hình tốt nhất = RMSE validation thấp nhất. Biến mục tiêu huấn luyện là `log1p(giá)` rồi đổi ngược về VND; `reports/target_transform_ablation.csv` đối chứng với dự đoán trực tiếp giá. Chia nhóm theo định nghĩa **bảo thủ** (không gồm km): `scripts/group_sensitivity.py` cho thấy định nghĩa lỏng hơn làm chỉ số test đẹp giả do rò rỉ tin đăng lại.

**Cổng triển khai mô hình:** bản mới nằm ở `models/versions/<phiên bản>/`; chỉ khi nạp được, dự đoán hữu hạn và dương, và tốt hơn mức tham chiếu (trung vị giá) thì mới thay `models/*.joblib`, và bản cũ được lưu vào `models/archive/<phiên bản cũ>/` trước.

## 7. Kết quả (đọc ở đâu)
| File | Nội dung |
|---|---|
| `reports/supervised_results.csv` | Mọi (tỷ lệ × mô hình): CV-RMSE, RMSE validation baseline/tuned, MAE/RMSE/R²/MAPE/medAPE trên test |
| `reports/model_comparison_table.md/.csv` | Bảng so sánh ở tỷ lệ phục vụ + mức tham chiếu "luôn đoán trung vị" |
| `reports/feature_importance.csv/.png` | Permutation importance theo cột và theo nhóm cột tương quan |
| `reports/performance_by_group.csv` | Sai số theo hãng, kiểu dáng, nhiên liệu, hộp số, tuổi xe, phân khúc giá, có/không có km, nguồn |
| `reports/worst_errors.csv` | Các ca sai tuyệt đối/tương đối lớn nhất để phân tích nguyên nhân |
| `reports/target_transform_ablation.csv`, `group_sensitivity.csv` | Đối chứng biến mục tiêu; độ nhạy với cách chia nhóm |
| `reports/BAO_CAO_NANG_CAP.md` | Báo cáo tổng hợp tự sinh |

Cách đọc chỉ số: MAE/RMSE bằng VND (RMSE nhạy với xe giá rất cao); R² gần 1 là tốt; medAPE = sai số tương đối trung vị. Không có tuyên bố về độ chính xác nào ngoài những gì các file trên cho thấy trên **tập test giữ riêng, cùng nguồn và cùng thời điểm thu thập**; độ chính xác với tin rao mới/giá hiện nay chưa được kiểm chứng.

## 8. Học không giám sát và bán giám sát
- `unsupervised.py`: PCA (2 chiều), K-Means (k chọn theo silhouette, kèm elbow/Davies-Bouldin), DBSCAN (`eps` chọn bằng lưới phân vị). Fit chỉ trên **train**, **không dùng `price`**; giá chỉ dùng để mô tả cụm sau đó. Silhouette tính trên mẫu 5.000 điểm, DBSCAN chạy trên mẫu 8.000 điểm cố định seed (tính toàn bộ cần hàng GB RAM).
- `semi_supervised.py`: ẩn có kiểm soát một phần nhãn train, so baseline (chỉ nhãn có) – self-training (pseudo-label theo độ bất định giữa các cây) – oracle (đủ nhãn); nhãn test không bị đụng. Pseudo-label có thể khuếch đại sai số nên kết quả có thể **kém hơn** baseline; hãy đọc `reports/semi_supervised_summary.csv`.

## 9. MLflow
```powershell
python src\train_supervised.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000      # hoặc .\scripts\start_mlflow.ps1
```
Mở http://127.0.0.1:5000, experiment `car-price-prediction`. Mỗi (mô hình × tỷ lệ) là một run: tham số (thuật toán, chiến lược chia, số dòng train/val/test, `data_version`, danh sách đặc trưng, siêu tham số), chỉ số (CV-RMSE, validation baseline/tuned, test MAE/RMSE/R²/MAPE/medAPE), artifact (`data_info.json`, `cleaning_report.json`) và mô hình (Pipeline, định dạng cloudpickle). Lần chạy cũ trên dữ liệu mô phỏng được lưu ở `archive/mlflow_demo_simulated/`.

## 10. API và web
```powershell
uvicorn app.main:app --reload       # hoặc .\scripts\start_api.ps1
```
Web: http://127.0.0.1:8000 (giao diện React nếu đã `npm run build`, nếu chưa thì tự dùng giao diện HTML cũ; giao diện cũ luôn ở `/cu`) · Tài liệu API: http://127.0.0.1:8000/docs

Nếu Windows chặn `uvicorn.exe` (Application Control), chạy bằng `python -m uvicorn app.main:app --reload`.

| Endpoint | Chức năng |
|---|---|
| `GET /health` | Trạng thái; 503 nếu chưa có mô hình |
| `GET /api/meta` | Danh sách trường (nhãn, bắt buộc/tùy chọn, giá trị hợp lệ), mô hình, đơn vị, phiên bản |
| `GET /api/metrics` · `/api/metrics/all` | Chỉ số mô hình đang phục vụ · toàn bộ thí nghiệm |
| `POST /api/predict` | `{"values": {...}, "model": "random_forest"}`; trả giá VND, giá của mọi mô hình trên cùng đầu vào, `warnings`, `defaults_used` |
| `POST /api/feedback` | Giá thực tế đến sau `{"prediction_id","actual_price"}` |
| `GET /api/monitoring` | Số yêu cầu, độ trễ, lỗi, data drift, sai số theo nhãn thực tế |
| `GET /api/model-info` | **Mới, thêm vào, không đổi endpoint cũ.** Phiên bản mô hình, thời điểm huấn luyện, phiên bản dữ liệu, số dòng train/val/test, cách chia, tinh chỉnh, thư viện; dùng cho khối "Kho mô hình" |

Ví dụ: `curl -X POST http://127.0.0.1:8000/api/predict -H "Content-Type: application/json" -d "{\"values\":{\"brand\":\"toyota\",\"model\":\"vios\",\"year\":2018,\"mileage_km\":60000}}"`

Chỉ **`brand`, `model`, `year`** là bắt buộc; trường tùy chọn bỏ trống được xử lý như dữ liệu thiếu (dung tích động cơ và số chỗ được điền bằng trung vị của chính dòng xe, học từ train; các trường phân loại thiếu được xử lý như nhóm "unknown", giống lúc huấn luyện). Trả **422** kèm danh sách lỗi khi: thiếu trường bắt buộc, sai kiểu (kể cả `true/false`), năm/km/dung tích/số chỗ ngoài phạm vi hợp lệ, hãng hoặc giá trị phân loại chưa từng thấy, trường lạ, tên mô hình sai. Dòng xe/phiên bản chưa gặp chỉ **cảnh báo** (`warnings`); giá trị ngoài khoảng huấn luyện cũng cảnh báo là kém tin cậy.

### 10.1. Giao diện web mới (`frontend/`)
Công nghệ: React 19, TypeScript, Vite, Tailwind CSS 4, Recharts, lucide-react, font Be Vietnam Pro (tự lưu trong gói, chạy được khi không có mạng). Toàn bộ chữ hiển thị bằng tiếng Việt.

| Trang | Đường dẫn | Dữ liệu lấy từ |
|---|---|---|
| Trang chủ | `/` | `/api/metrics`, `/api/meta` (số liệu thật, không nhập tay) |
| Định giá ô tô | `/dinh-gia` | Biểu mẫu sinh từ `/api/meta`; gửi `POST /api/predict`; gửi giá thực tế qua `POST /api/feedback` |
| Gợi ý xe phù hợp | `/goi-y` | **Chưa tích hợp** (xem dưới) |
| Chi tiết xe | `/xe/:id` | **Chưa tích hợp** (xem dưới) |
| Lịch sử dự đoán | `/lich-su` | Lưu trong trình duyệt (localStorage) từ kết quả API thật, tối đa 50 lần, có nút xóa kèm xác nhận |
| Quản trị hệ thống | `/quan-tri` | `/health`, `/api/metrics`, `/api/metrics/all`, `/api/monitoring`, `/api/model-info` |

Lớp gọi API nằm ở `frontend/src/services/` (`predictionService.ts`, `monitoringService.ts`, `recommendationService.ts`, `capabilityService.ts`, `http.ts`). Biểu mẫu chỉ có đúng các trường backend hỗ trợ (hãng, dòng, phiên bản, kiểu dáng, năm, km, dung tích, hộp số, nhiên liệu, dẫn động, số chỗ, xuất xứ, tỉnh/thành); trường bắt buộc là hãng, dòng, năm. Giá không bao giờ được tính ở frontend.

**Chạy khi phát triển** (hai cửa sổ PowerShell):
```powershell
# Cửa sổ 1: backend
python -m uvicorn app.main:app --reload
# Cửa sổ 2: frontend (cần Node.js 20 trở lên)
cd frontend
npm install
npm run dev          # http://localhost:5173 ; /api được chuyển tiếp tới http://127.0.0.1:8000
```
**Chạy bản build (một cổng duy nhất):**
```powershell
cd frontend; npm install; npm run build; cd ..
python -m uvicorn app.main:app      # http://127.0.0.1:8000 phục vụ cả API lẫn giao diện
```
Biến môi trường (xem `frontend/.env.example`): `VITE_BACKEND_URL` (địa chỉ FastAPI cho proxy khi dev), `VITE_API_BASE` (khi tách địa chỉ, cần bật CORS), `VITE_CONTACT_EMAIL`, `VITE_CONTACT_PHONE`, `VITE_REPO_URL` (chân trang; để trống thì không hiển thị, không bịa thông tin liên hệ). Kiểm tra mã: `npm run lint`, `npm run build`.

**Chức năng chưa tích hợp (trung thực):** backend hiện **chưa có** API gợi ý xe, danh sách xe hay chi tiết xe, và dự án cũng chưa có kho ảnh xe có nguồn rõ ràng. Hai trang `/goi-y` và `/xe/:id` tự kiểm tra `/openapi.json`; khi chưa có endpoint, trang hiển thị "chưa tích hợp" và không hiển thị xe giả. Khi backend bổ sung đúng hợp đồng dưới đây, trang sẽ tự nhận và hiển thị kết quả, không cần sửa lại giao diện. Hợp đồng đề xuất (mới, không ảnh hưởng API cũ):

| Endpoint (đề xuất) | Yêu cầu | Phản hồi |
|---|---|---|
| `POST /api/recommend` | `{ "budget_min", "budget_max" (VND), "brand", "model", "year_from", "fuel_type", "transmission", "body_type", "method": "knn" \| "content_based", "limit" }`; các bộ lọc đều tùy chọn | `{ "items": [CarListing], "method" }` |
| `GET /api/cars/{id}` | không | `CarListing` |

`CarListing`: `id`, `brand`, `model`, `variant?`, `year`, `mileage_km?`, `engine_capacity?`, `fuel_type?`, `transmission?`, `body_type?`, `origin?`, `province?`, `seats?`, `price` (VND), `image_url?`, `highlights?: string[]`, `similar_ids?: string[]` (dùng cho phần so sánh xe tương tự).

Ảnh chỉ hiển thị khi có `image_url` (backend chỉ nên trả ảnh có nguồn rõ ràng); không có thì hiện khung "Chưa có ảnh". Không gắn ảnh ngẫu nhiên vào một mẫu xe cụ thể.

**Hạn chế:** trang Quản trị không có đăng nhập vì backend hiện không có xác thực (nên không công khai ra Internet khi chưa thêm xác thực). Kho mô hình hiển thị phiên bản đang phục vụ; MLflow Model Registry (đăng ký, chuyển giai đoạn) chưa được tích hợp. PSI chỉ có ý nghĩa khi đã có từ 30 yêu cầu dự đoán.

## 11. Giám sát
Log ở `logs/` (`predictions.jsonl`, `errors.jsonl`, `feedback.jsonl`). **Data drift** đo bằng PSI (>0,2 là dấu hiệu, cần ≥30 yêu cầu; cột nhiều giá trị được gộp 15 giá trị phổ biến + "khác"; trường bỏ trống không tính). **Concept drift** chỉ đánh giá được khi có nhãn thực tế qua `/api/feedback`; data drift không đồng nghĩa concept drift.

## 12. Docker, kiểm thử, CI
```powershell
python src\train_supervised.py                    # cần có models\ trước khi build
docker build -t car-price-api .                    # build nhiều giai đoạn: Node build frontend, rồi đóng vào ảnh Python
docker compose up --build                          # API (8000) + MLflow UI (5000)
python -m pytest -q                                # kiểm thử (dùng python -m để đúng môi trường)
ruff check src app tests --select E9,F --ignore F401
```
`tests/` kiểm tra: dữ liệu thật (≥10.000 dòng, giá/năm/km hợp lệ, không trùng, báo cáo làm sạch khớp), quy tắc làm sạch trên bảng dựng tay, chia nhóm không rò rỉ, đặc trưng phái sinh, huấn luyện 3 mô hình, chọn theo validation, lưu/nạp mô hình, cổng triển khai, lưu bản cũ, API (thiếu/sai/ngoài phạm vi), MLflow (tham số, chỉ số, artifact, mô hình). Các test huấn luyện dùng mẫu con 3.000 dòng thật và thư mục tạm, không đụng `models/` và `reports/` thật. CI chạy ruff, import, tải/đối chiếu dữ liệu và pytest; job `frontend` chạy `npm ci`, ESLint và build.

## 13. Hạn chế
- Giá rao ≠ giá giao dịch; dữ liệu ~2022 nên không phản ánh thị trường hiện nay; không dùng để ra quyết định mua bán thực.
- `seats` của bonbanh dùng cả 4 và 5 cho sedan (cách gọi "xe 4 chỗ"), còn chotot dùng 5 → cột này có phần là dấu hiệu nguồn.
- bonbanh: ~45% tin không có km (coi là thiếu); chotot: không có phiên bản, dẫn động; nguồn không có ngày đăng/mã lực.
- Tên dòng xe hai nguồn không hoàn toàn thống nhất (chotot chỉ giữ từ đầu, vd `cx` so với `cx5`); `province` thiếu với toàn bộ bonbanh nên mô hình có thể dùng "thiếu tỉnh" như dấu hiệu nguồn.
- Không phát hiện được cùng một xe đăng ở cả hai website; có thể còn rò rỉ nhỏ giữa các tin đăng lại có thuộc tính khác nhau.
- Xe hiếm/siêu sang rất ít mẫu → sai số tuyệt đối lớn (xem `performance_by_group.csv`, `worst_errors.csv`).
- Docker image chưa được build thử trong môi trường tạo gói (không có Docker daemon); MLflow UI trong container chỉ xem được tham số/chỉ số, artifact có thể cần chạy trên máy đã huấn luyện (đường dẫn tuyệt đối).
- Chưa dùng SHAP (chỉ permutation importance).

## 14. Lỗi thường gặp
| Lỗi | Cách xử lý |
|---|---|
| `Không thấy dữ liệu trong Data/raw` | `python scripts\download_data.py` (hoặc tải tay 2 file theo `Data/raw/SOURCES.md`) |
| `ModuleNotFoundError: xgboost/mlflow/fastapi` | `pip install -r requirements-dev.txt` |
| API trả 503 "Thiếu models/..." | `python src\train_supervised.py` |
| "Lược đồ đặc trưng của mô hình đã lưu khác config" | Mô hình cũ (dữ liệu mô phỏng): huấn luyện lại |
| `pytest` báo thiếu thư viện | Dùng `python -m pytest` trong đúng virtualenv |
| MLflow UI trống | Chạy từ thư mục gốc, dùng `--backend-store-uri sqlite:///mlflow.db` |
| `Activate.ps1` bị chặn | `Set-ExecutionPolicy -Scope Process Bypass` |
| Hết bộ nhớ khi chạy unsupervised | Giảm `SIL_N`/`DB_N` trong `src/unsupervised.py` |
