import numpy as np
import pandas as pd

import config as C
from monitoring import Monitor, build_reference, drift_report


def _ref():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"a": rng.normal(100, 10, 500), "c": rng.choice(["x", "y"], 500, p=[.7, .3])})
    return X, build_reference(X)


def test_no_drift_same_distribution():
    X, ref = _ref()
    rng = np.random.RandomState(1)
    rows = [{"a": rng.normal(100, 10), "c": rng.choice(["x", "y"], p=[.7, .3])} for _ in range(300)]
    assert drift_report(ref, rows)["drifted_features"] == []


def test_drift_detected_when_shifted():
    _, ref = _ref()
    rng = np.random.RandomState(2)
    rows = [{"a": rng.normal(140, 10), "c": "y"} for _ in range(100)]
    rep = drift_report(ref, rows)
    assert set(rep["drifted_features"]) == {"a", "c"}
    assert "concept" in rep["note"].lower()


def test_insufficient_samples():
    _, ref = _ref()
    assert drift_report(ref, [{"a": 1, "c": "x"}] * 5)["status"] == "insufficient_samples"


def test_logs_and_feedback(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "LOGS", tmp_path)   # tách log riêng, không lẫn với test khác
    m = Monitor()
    m.record_prediction("p1", "random_forest", "v1", {"a": 1}, 1000.0, 5.0)
    m.record_error("/api/predict", 422, "bad")
    s = m.summary(None, "v1")
    assert s["n_predictions"] == 1 and s["n_errors"] == 1
    assert s["feedback"]["n"] == 0 and "chưa" in s["feedback"]["message"].lower()
    m.record_feedback("p1", 1200.0)
    f = m.summary(None, "v1")["feedback"]
    assert f["n"] == 1 and f["MAE"] == 200.0
    assert (tmp_path / "predictions.jsonl").exists()