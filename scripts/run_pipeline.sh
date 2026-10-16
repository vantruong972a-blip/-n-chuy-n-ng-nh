#!/usr/bin/env bash
set -e
python -m pip install -r requirements-dev.txt
python src/preprocess.py
python src/train_supervised.py --all
python src/unsupervised.py
python src/semi_supervised.py
