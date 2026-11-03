#!/usr/bin/env bash
# bash scripts/run_pipeline.sh [fast|all]
set -e
python -m pip install -r requirements-dev.txt
python scripts/download_data.py
python src/preprocess.py
case "$1" in
  fast) python src/train_supervised.py --fast --max-rows 4000 --no-mlflow ;;
  all)  python src/train_supervised.py --all ;;
  *)    python src/train_supervised.py ;;
esac
python src/unsupervised.py
if [ "$1" = "fast" ]; then python src/semi_supervised.py --max-rows 4000 --seeds 1 --hide 0.5; else python src/semi_supervised.py; fi
python scripts/make_report.py
