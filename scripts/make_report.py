"""Tạo reports/BAO_CAO_NANG_CAP.md TỪ các file kết quả thật (không gõ tay số liệu).
Chạy sau khi đã chạy preprocess.py và train_supervised.py:  python scripts/make_report.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402

R, P = C.REPORTS, C.PROCESSED


def tr(x):
    return "-" if pd.isna(x) else f"{x / 1e6:,.1f}"


def main():
    rep = json.loads((P / "cleaning_report.json").read_text(encoding="utf-8"))
    res = pd.read_csv(R / "supervised_results.csv")
    info = json.loads((C.MODELS / "model_info.json").read_text(encoding="utf-8")) if (C.MODELS / "model_info.json").exists() else {}
    L = ["# Báo cáo nâng cấp dự án (sinh tự động từ file kết quả)\n"]
    L.append("## 1. Dữ liệu\n")
    L.append(f"- Bản ghi thô: **{rep['n_raw']:,}** ({', '.join(f'{k} {v:,}' for k, v in rep['n_raw_by_source'].items())}).")
    L.append(f"- Sau làm sạch (có nhãn giá): **{rep['n_labeled']:,}** ({', '.join(f'{k} {v:,}' for k, v in rep['n_labeled_by_source'].items())}); "
             f"tập chưa nhãn: {rep['n_unlabeled']:,}; số nhóm tin rao (group_id): {rep['n_groups']:,}.")
    old = R / "archive_demo_simulated" / "supervised_results.csv"
    L.append("- Trước nâng cấp: dữ liệu MÔ PHỎNG 205 dòng (đã chuyển sang `Data/demo/`, không dùng để huấn luyện).\n")
    L.append("| Bước làm sạch | Loại | Còn lại |\n|---|---:|---:|")
    for s in rep["steps"]:
        L.append(f"| {s['step']} | {s['removed']:,} | {s['remaining']:,} |")
    L.append("\nSửa mức trường (giữ dòng, đặt về thiếu): " + ", ".join(f"{k}={v:,}" for k, v in rep["field_fixes"].items()))
    L.append("\nLý do loại dòng: " + ", ".join(f"{k}={v:,}" for k, v in rep["rejected_by_reason"].items()))
    L.append(f"\nGiá (VND) sau làm sạch: trung vị {rep['price_vnd']['50%'] / 1e6:,.0f} triệu; p1 {rep['price_vnd']['1%'] / 1e6:,.0f} triệu; "
             f"p99 {rep['price_vnd']['99%'] / 1e9:,.2f} tỷ; max {rep['price_vnd']['max'] / 1e9:,.1f} tỷ.\n")
    L.append("## 2. Kết quả mô hình (tập test giữ riêng, chia theo nhóm)\n")
    md = R / "model_comparison_table.md"
    L.append(md.read_text(encoding="utf-8") if md.exists() else "(chưa có)")
    L.append("\nTất cả tỷ lệ chia đã chạy:\n")
    t = res[["split", "model", "config_used", "val_RMSE", "MAE", "RMSE", "R2", "medAPE", "selected_best"]].copy()
    L.append("| Tỷ lệ | Mô hình | Cấu hình | Val RMSE (tr) | Test MAE (tr) | Test RMSE (tr) | R² | medAPE % | Chọn |\n|---|---|---|---:|---:|---:|---:|---:|:-:|")
    for _, r in t.iterrows():
        L.append(f"| {r.split} | {r.model} | {r.config_used} | {tr(r.val_RMSE)} | {tr(r.MAE)} | {tr(r.RMSE)} | {r.R2:.3f} | {r.medAPE:.1f} | {'✔' if r.selected_best else ''} |")
    if info:
        L.append(f"\nMô hình phục vụ API: **{info['best_model']}** (phiên bản `{info['model_version']}`, dữ liệu `{info['data_version']}`).")
    if old.exists():
        o = pd.read_csv(old)
        o = o[o.split == "80/20"]
        L.append("\n### So sánh trước/sau (CHỈ mang tính tham khảo)\n")
        L.append("Kết quả cũ trên dữ liệu mô phỏng (USD, test 41 dòng) KHÔNG so sánh trực tiếp được với kết quả mới (VND, dữ liệu thật). "
                 "Chỉ R² là không phụ thuộc đơn vị, nhưng vẫn là hai bài toán khác nhau.\n")
        L.append("| Mô hình | R² cũ (mô phỏng, 80/20) | R² mới (thật, 80/20) |\n|---|---:|---:|")
        for _, r in o.iterrows():
            n = res[(res.model == r.model) & (res.split == "80/20")]
            L.append(f"| {r.model} | {r.R2:.3f} | {'-' if n.empty else format(n.R2.iloc[0], '.3f')} |")
    ab = R / "target_transform_ablation.csv"
    if ab.exists():
        a = pd.read_csv(ab)
        L.append("\n### Đối chứng biến mục tiêu (validation, tham số mặc định)\n")
        L.append("| Mô hình | Mục tiêu | MAE (tr) | RMSE (tr) | R² | medAPE % |\n|---|---|---:|---:|---:|---:|")
        for _, r in a.iterrows():
            L.append(f"| {r.model} | {r.target} | {tr(r.MAE)} | {tr(r.RMSE)} | {r.R2:.3f} | {r.medAPE:.1f} |")
    imp = R / "feature_importance.csv"
    if imp.exists() and info:
        i = pd.read_csv(imp)
        i = i[i.model == info["best_model"]].head(12)
        L.append(f"\n## 3. Permutation importance ({info['best_model']}, test)\n")
        L.append("Xáo trộn một cột (hoặc cả nhóm cột tương quan, dòng [nhóm]) rồi đo RMSE tăng thêm. Các cột tương quan (hãng/dòng/phiên bản) chia sẻ vai trò nên xem cả dòng nhóm; đây là mức phụ thuộc của mô hình, không phải quan hệ nhân quả.\n")
        L.append("| Đặc trưng | RMSE tăng khi xáo (triệu VND) |\n|---|---:|")
        for _, r in i.iterrows():
            L.append(f"| {r.feature} | {tr(r.rmse_increase_vnd)} |")
    g = R / "performance_by_group.csv"
    if g.exists() and info:
        d = pd.read_csv(g)
        d = d[(d.model == info["best_model"]) & d.group_by.isin(["price_band", "age_band", "mileage_known", "source"])]
        L.append("\n## 4. Hiệu năng theo nhóm (mô hình tốt nhất, test)\n")
        L.append("| Nhóm theo | Giá trị | n | MAE (tr) | RMSE (tr) | R² | medAPE % |\n|---|---|---:|---:|---:|---:|---:|")
        for _, r in d.iterrows():
            L.append(f"| {r.group_by} | {r.group} | {r.n} | {tr(r.MAE)} | {tr(r.RMSE)} | {r.R2:.3f} | {r.medAPE:.1f} |")
    w = R / "worst_errors.csv"
    if w.exists():
        x = pd.read_csv(w)
        x = x[x.rank_by == "abs_err_vnd"].head(10)
        L.append("\n## 5. 10 ca sai tuyệt đối lớn nhất (test)\n")
        L.append("| Nguồn | Hãng | Dòng | Năm | Km | Giá thực (tr) | Dự đoán (tr) |\n|---|---|---|---:|---:|---:|---:|")
        for _, r in x.iterrows():
            L.append(f"| {r.source} | {r.brand} | {r.model} | {r.year:.0f} | {'-' if pd.isna(r.mileage_km) else format(r.mileage_km, ',.0f')} | {tr(r.price)} | {tr(r.pred)} |")
    L.append("\n## 6. Hạn chế\n")
    L += ["- Giá là **giá niêm yết** (người bán rao) thu thập khoảng 2022, không phải giá giao dịch; không phản ánh giá hiện tại.",
          "- Không có ngày đăng, mã lực; `variant` chỉ có ở bonbanh, `province` chỉ có ở chotot (giá trị thiếu được mô hình xử lý như một nhóm).",
          f"- bonbanh ghi km = 0 cho {rep['field_fixes']['mileage_km_zero_or_missing_to_nan']:,} tin, được coi là 'chưa khai báo' (thiếu), không phải xe chạy 0 km.",
          "- Tên dòng xe không hoàn toàn thống nhất giữa 2 nguồn (vd chotot chỉ giữ từ đầu: `cx` so với `cx5`).",
          "- Không phát hiện được cùng một xe đăng trên cả hai website; tin đăng lặp trong cùng nguồn được gom nhóm khi chia dữ liệu.",
          "- Xe hiếm/siêu sang có rất ít mẫu nên sai số lớn (vd Rolls-Royce, Maybach); ba mô hình có thể lệch nhau hàng chục tỷ VND cho các xe này. R² trong từng phân khúc giá thấp một phần vì phương sai giá trong mỗi dải hẹp, hãy đọc cùng medAPE.",
          "- Một số giá nghi sai vẫn còn vì không đủ bằng chứng để loại theo quy tắc (vd Audi Q7 2012 giá 9,5 tỷ trên chotot, chỉ lệch ~10 lần trung vị dòng xe < ngưỡng 15 lần); chúng làm tăng sai số test.",
          "- Mô hình tốt nhất đổi giữa các tỷ lệ chia (chọn theo validation); chênh lệch RF/XGBoost ở 80/20 rất nhỏ (xem bảng), không nên coi một mô hình là vượt trội chắc chắn.",
          "- Dữ liệu do bên thứ ba crawl; kiểm tra điều khoản sử dụng và giấy phép GPL-3.0 của repo gốc khi dùng ngoài mục đích học tập."]
    (R / "BAO_CAO_NANG_CAP.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("Đã ghi", R / "BAO_CAO_NANG_CAP.md")


if __name__ == "__main__":
    main()
