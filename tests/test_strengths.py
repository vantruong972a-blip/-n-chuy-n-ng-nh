"""/api/model-strengths: chỉ đọc reports/model_roles.csv, không bịa số."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _write(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "segment_type", "segment", "n", "RMSLE", "medAPE", "bias_pct"])
        w.writerows(rows)


@pytest.fixture
def client_tmp(tmp_path, monkeypatch):
    import config as C
    from app import main
    monkeypatch.setattr(C, "REPORTS", tmp_path)
    return TestClient(main.app), tmp_path


def test_missing_file_is_reported_not_faked(client_tmp):
    c, _ = client_tmp
    r = c.get("/api/model-strengths")
    assert r.status_code == 200
    assert r.json()["available"] is False and r.json()["segments"] == []


def test_best_and_close_are_computed_from_file(client_tmp):
    c, d = client_tmp
    _write(d / "model_roles.csv", [
        ["linear_regression", "tuổi xe", "0-2 năm", 100, 0.30, 7.0, 0],
        ["random_forest", "tuổi xe", "0-2 năm", 100, 0.20, 5.0, 0],
        ["xgboost", "tuổi xe", "0-2 năm", 100, 0.1, 4.0, 0],
        ["linear_regression", "số km", "có khai báo km", 50, 0.25, 7.0, 0],
        ["random_forest", "số km", "có khai báo km", 50, 0.2400, 6.0, 0],
        ["xgboost", "số km", "có khai báo km", 50, 0.2410, 6.0, 0],
    ])
    j = c.get("/api/model-strengths").json()
    assert j["available"] is True and len(j["segments"]) == 2
    by = {s["segment"]: s for s in j["segments"]}
    assert by["0-2 năm"]["best"] == "xgboost" and by["0-2 năm"]["close"] is False
    assert by["có khai báo km"]["best"] == "random_forest" and by["có khai báo km"]["close"] is True
    assert by["0-2 năm"]["n"] == 100 and by["0-2 năm"]["models"]["xgboost"]["rmsle"] == 0.1


def test_bad_rows_are_skipped(client_tmp):
    c, d = client_tmp
    _write(d / "model_roles.csv", [["xgboost", "tuổi xe", "0-2 năm", "abc", 0.1, 4.0, 0], ["xgboost", "tuổi xe", "3-5 năm", 10, 0.1, 4.0, 0]])
    j = c.get("/api/model-strengths").json()
    assert [s["segment"] for s in j["segments"]] == ["3-5 năm"]
