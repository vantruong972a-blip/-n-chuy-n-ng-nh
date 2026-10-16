# Chạy toàn bộ pipeline trên Windows PowerShell (chạy từ thư mục gốc project)
#   .\scripts\run_pipeline.ps1            # đầy đủ (4 tỷ lệ, có tinh chỉnh)
#   .\scripts\run_pipeline.ps1 -Fast      # chạy thử nhanh
param([switch]$Fast)
$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$fast = if ($Fast) { "--fast" } else { "" }
.\.venv\Scripts\python.exe src\preprocess.py
.\.venv\Scripts\python.exe src\train_supervised.py --all $fast
.\.venv\Scripts\python.exe src\unsupervised.py
.\.venv\Scripts\python.exe src\semi_supervised.py
Write-Host "Xong. Mở MLflow: .\scripts\start_mlflow.ps1   |   Mở API: .\scripts\start_api.ps1"
