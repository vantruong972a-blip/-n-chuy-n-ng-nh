"""So sánh mô hình ỨNG VIÊN (bản mới huấn luyện) với mô hình ĐANG PHỤC VỤ API, trên CÙNG một tập validation.

    python scripts/compare_candidate.py                 # ứng viên = bản mới nhất trong models/versions/
    python scripts/compare_candidate.py --candidate 20261110-053813-d5e70f1

Chỉ ĐỀ XUẤT, không tự thay mô hình. Muốn nâng cấp: chạy lại `python src/train_supervised.py` (tự lưu bản cũ vào
models/archive/) hoặc `python scripts/register_champion.py` sau khi đã quyết định.

Lưu ý khi đọc kết quả: cả hai mô hình được chấm trên tập validation của DỮ LIỆU HIỆN TẠI. Mô hình đang phục vụ có thể
đã thấy một số dòng này lúc huấn luyện (nếu dữ liệu cũ nằm trong train cũ), nên so sánh hơi có lợi cho nó; vì vậy cần cải
thiện RMSLE vượt ngưỡng `--margin` mới đề xuất thay.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib  # noqa: E402

import config as C  # noqa: E402
import evaluation as E  # noqa: E402
from preprocess import get_split3  # noqa: E402


def decide(champ_rmsle, cand_rmsle, margin=0.005):
    """Đề xuất thay khi RMSLE của ứng viên thấp hơn mô hình đang chạy ít nhất `margin` (tuyệt đối)."""
    if cand_rmsle <= champ_rmsle - margin:
        return "thay", "ứng viên tốt hơn đáng kể (RMSLE thấp hơn quá ngưỡng)"
    if cand_rmsle <= champ_rmsle:
        return "giu", "ứng viên chỉ nhỉnh hơn không đáng kể, giữ mô hình hiện tại"
    return "giu", "ứng viên không tốt hơn, giữ mô hình hiện tại"


def _latest_version():
    vs = sorted(p for p in (C.MODELS / "versions").glob("*") if p.is_dir())
    return vs[-1].name if vs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", help="tên thư mục trong models/versions/ (mặc định: bản mới nhất)")
    ap.add_argument("--margin", type=float, default=0.005, help="mức cải thiện RMSLE tối thiểu để đề xuất thay")
    a = ap.parse_args()

    cur_info = C.MODELS / "model_info.json"
    ver = a.candidate or _latest_version()
    if not ver or not (C.MODELS / "versions" / ver / "model_info.json").exists():
        raise SystemExit("Không có bản ứng viên trong models/versions/. Hãy huấn luyện trước: python src/train_supervised.py --no-promote")
    cand_info = json.loads((C.MODELS / "versions" / ver / "model_info.json").read_text(encoding="utf-8"))
    cand_name = cand_info["best_model"]
    cand = joblib.load(C.MODELS / "versions" / ver / f"{cand_name}.joblib")

    sp = get_split3()
    X_va, y_va, _ = sp["val"]
    rows = {"candidate": {"version": ver, "model": cand_name, **E.metrics(y_va, cand.predict(X_va))}}

    if not cur_info.exists():
        print("Chưa có mô hình đang phục vụ (models/model_info.json) để so sánh. Ứng viên:", rows["candidate"])
        return
    info = json.loads(cur_info.read_text(encoding="utf-8"))
    if info["model_version"] == ver:
        raise SystemExit("Ứng viên chính là mô hình đang phục vụ; không có gì để so sánh.")
    champ = joblib.load(C.MODELS / f"{info['best_model']}.joblib")
    rows["champion"] = {"version": info["model_version"], "model": info["best_model"], **E.metrics(y_va, champ.predict(X_va))}

    verdict, why = decide(rows["champion"]["RMSLE"], rows["candidate"]["RMSLE"], a.margin)
    out = {"n_validation": int(len(y_va)), "margin": a.margin, "verdict": verdict, "reason": why, **rows}
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    (C.REPORTS / "candidate_vs_champion.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Tập validation hiện tại: {len(y_va)} xe\n")
    print(f"{'':10}{'phiên bản':26}{'mô hình':16}{'RMSLE':>8}{'MAE':>12}{'R2':>8}")
    for k, lab in (("champion", "đang chạy"), ("candidate", "ứng viên")):
        r = rows[k]
        print(f"{lab:10}{r['version']:26}{r['model']:16}{r['RMSLE']:8.4f}{r['MAE']:12.0f}{r['R2']:8.3f}")
    print(f"\nĐề xuất: {'THAY' if verdict == 'thay' else 'GIỮ'} - {why}.")
    print("Script này chỉ đề xuất, không tự thay mô hình. Đã ghi reports/candidate_vs_champion.json")


if __name__ == "__main__":
    main()
