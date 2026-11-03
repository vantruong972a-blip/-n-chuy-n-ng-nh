# Chạy toàn bộ pipeline trên Windows PowerShell (chạy từ thư mục gốc project)
#   .\scripts\run_pipeline.ps1                # đầy đủ, tỷ lệ phục vụ API 80/20 (có tinh chỉnh; vài chục phút)
#   .\scripts\run_pipeline.ps1 -AllSplits     # thêm 3 tỷ lệ 60/40, 70/30, 90/10 (rất lâu)
#   .\scripts\run_pipeline.ps1 -Fast          # chạy thử nhanh trên mẫu nhỏ (KHÔNG thay mô hình đang phục vụ)
param([switch]$Fast, [switch]$AllSplits)
$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r requirements-dev.txt
& $py scripts\download_data.py            # kiểm tra/tải dữ liệu thật vào Data\raw (đối chiếu SHA-256)
& $py src\preprocess.py
if ($Fast) { & $py src\train_supervised.py --fast --max-rows 4000 --no-mlflow }
elseif ($AllSplits) { & $py src\train_supervised.py --all }
else { & $py src\train_supervised.py }
& $py src\unsupervised.py
if ($Fast) { & $py src\semi_supervised.py --max-rows 4000 --seeds 1 --hide 0.5 } else { & $py src\semi_supervised.py }
& $py scripts\make_report.py
Write-Host "Xong. Mở MLflow: .\scripts\start_mlflow.ps1   |   Mở API: .\scripts\start_api.ps1"
