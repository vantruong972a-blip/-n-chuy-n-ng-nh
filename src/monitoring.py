"""Giám sát cơ bản: log dự đoán, độ trễ, lỗi, data drift (PSI) và sai số khi có nhãn thực tế.

Lưu ý khái niệm:
- DATA DRIFT: phân phối đầu vào thay đổi so với dữ liệu train. Tính được ngay từ log đầu vào.
- CONCEPT DRIFT: quan hệ giữa đầu vào và giá thay đổi. Chỉ đánh giá được khi có NHÃN THỰC TẾ
  đến sau (POST /api/feedback). Module này KHÔNG kết luận concept drift chỉ từ data drift.
"""
import json
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C

PSI_THRESHOLD = 0.2  # quy ước phổ biến: <0.1 ổn định, 0.1-0.2 thay đổi nhẹ, >0.2 thay đổi đáng kể
MIN_SAMPLES = 30


# ---------- thống kê tham chiếu (tính từ tập TRAIN) ----------
def build_reference(X: pd.DataFrame) -> dict:
    ref = {"n": int(len(X)), "numeric": {}, "categorical": {}}
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            s = s.dropna().astype(float)
            edges = np.unique(np.quantile(s, np.linspace(0, 1, 11)))
            inner = edges[1:-1]
            frac = np.bincount(np.digitize(s, inner), minlength=len(inner) + 1) / len(s)
            ref["numeric"][c] = {"mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()),
                                 "max": float(s.max()), "median": float(s.median()),
                                 "inner_edges": inner.tolist(), "fractions": frac.tolist()}
        else:
            vc = s.dropna().astype(str).value_counts(normalize=True)
            ref["categorical"][c] = {"freq": vc.to_dict(), "default": str(vc.index[0])}
    return ref


def _psi(expected, actual, eps=1e-4):
    e, a = np.clip(np.asarray(expected), eps, None), np.clip(np.asarray(actual), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(ref: dict, rows: list, min_samples=MIN_SAMPLES, threshold=PSI_THRESHOLD) -> dict:
    n = len(rows)
    out = {"n_samples": n, "threshold_psi": threshold, "features": {}, "drifted_features": [],
           "note": "Đây chỉ là dấu hiệu DATA drift (đầu vào đổi). Không đồng nghĩa với concept drift."}
    if n < min_samples:
        out["status"] = "insufficient_samples"
        out["message"] = f"Cần ít nhất {min_samples} yêu cầu dự đoán để tính drift (hiện có {n})."
        return out
    df = pd.DataFrame(rows)
    for c, r in ref["numeric"].items():
        if c not in df:
            continue
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(v) == 0:
            continue
        frac = np.bincount(np.digitize(v, r["inner_edges"]), minlength=len(r["inner_edges"]) + 1) / len(v)
        p = _psi(r["fractions"], frac)
        out["features"][c] = {"psi": round(p, 4), "flag": p > threshold, "type": "numeric"}
    for c, r in ref["categorical"].items():
        if c not in df:
            continue
        cats = sorted(set(r["freq"]) | set(df[c].astype(str)))
        e = [r["freq"].get(k, 0.0) for k in cats]
        a = df[c].astype(str).value_counts(normalize=True)
        p = _psi(e, [a.get(k, 0.0) for k in cats])
        out["features"][c] = {"psi": round(p, 4), "flag": p > threshold, "type": "categorical"}
    out["drifted_features"] = [k for k, v in out["features"].items() if v["flag"]]
    out["status"] = "data_drift_signal" if out["drifted_features"] else "no_signal"
    return out


# ---------- log yêu cầu ----------
class Monitor:
    def __init__(self):
        self._lock = threading.Lock()

    def _path(self, name):
        C.LOGS.mkdir(parents=True, exist_ok=True)
        return C.LOGS / name

    def _append(self, name, rec):
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **rec}
        with self._lock, open(self._path(name), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _read(self, name):
        p = self._path(name)
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def record_prediction(self, prediction_id, model, version, inputs, price, latency_ms):
        self._append("predictions.jsonl", {"id": prediction_id, "model": model, "model_version": version,
                                           "inputs": inputs, "price": price, "latency_ms": round(latency_ms, 2)})

    def record_error(self, endpoint, status, detail, latency_ms=None):
        self._append("errors.jsonl", {"endpoint": endpoint, "status": status, "detail": str(detail)[:500],
                                      "latency_ms": None if latency_ms is None else round(latency_ms, 2)})

    def record_feedback(self, prediction_id, actual_price):
        self._append("feedback.jsonl", {"id": prediction_id, "actual_price": float(actual_price)})

    def has_prediction(self, prediction_id):
        return any(r["id"] == prediction_id for r in self._read("predictions.jsonl"))

    def summary(self, ref=None, model_version=None):
        preds, errs, fb = (self._read(n) for n in ("predictions.jsonl", "errors.jsonl", "feedback.jsonl"))
        lat = [r["latency_ms"] for r in preds if r.get("latency_ms") is not None]
        res = {"model_version": model_version, "n_predictions": len(preds), "n_errors": len(errs),
               "latency_ms": {"mean": round(float(np.mean(lat)), 2) if lat else None,
                              "p95": round(float(np.percentile(lat, 95)), 2) if lat else None},
               "recent_errors": errs[-5:]}
        res["drift"] = drift_report(ref, [r["inputs"] for r in preds[-500:]]) if ref else {
            "status": "no_reference", "message": "Chưa có thống kê tham chiếu (chưa huấn luyện)."}
        res["feedback"] = self._feedback_metrics(preds, fb)
        return res

    @staticmethod
    def _feedback_metrics(preds, fb):
        by_id = {r["id"]: r for r in preds}
        rows = [{"ts": f["ts"], "err": by_id[f["id"]]["price"] - f["actual_price"], "actual": f["actual_price"]}
                for f in fb if f["id"] in by_id]
        if not rows:
            return {"n": 0, "message": "Chưa có nhãn thực tế gửi về nên chưa tính được sai số thực tế theo thời gian."}
        df = pd.DataFrame(rows)
        df["day"] = df["ts"].str[:10]
        daily = df.groupby("day")["err"].apply(lambda e: float(np.abs(e).mean())).round(2).to_dict()
        return {"n": len(df), "MAE": round(float(df["err"].abs().mean()), 2),
                "RMSE": round(float(np.sqrt((df["err"] ** 2).mean())), 2), "mae_by_day": daily}


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self.t) * 1000
