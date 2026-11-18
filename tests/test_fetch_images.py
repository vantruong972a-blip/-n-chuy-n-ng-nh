"""Kiểm thử script tải ảnh bằng phản hồi GIẢ có cùng cấu trúc với API của Wikimedia Commons (không gọi mạng)."""
import argparse
import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import fetch_car_images as F  # noqa: E402


def page(idx, title, w=2000, h=1200, lic="CC BY-SA 4.0", mime="image/jpeg", cats="Toyota Vios (XP150)", artist='<a href="x">Nguyen <b>An</b></a>'):
    return {"title": title, "index": idx, "imageinfo": [{
        "thumburl": f"https://upload.wikimedia.org/thumb/{idx}.jpg", "url": f"https://upload.wikimedia.org/{idx}.jpg",
        "descriptionurl": f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}", "mime": mime, "width": w, "height": h,
        "extmetadata": {"Artist": {"value": artist}, "LicenseShortName": {"value": lic},
                        "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"}, "Categories": {"value": cats}}}]}


def api(*pages):
    return {"query": {"pages": {str(i): p for i, p in enumerate(pages)}}}


def test_slug_query_and_tokens():
    assert F.slug("Vf E34") == "vf-e34" and F.slug("mercedes-benz") == "mercedes-benz"
    assert F.build_query("toyota", "vios") == "Toyota Vios filetype:bitmap"
    assert F.build_query("mercedes-benz", "c") == "Mercedes-Benz C filetype:bitmap"
    assert F.tokens("File:Toyota Vios 2019.jpg") == ["file", "toyota", "vios", "2019", "jpg"]


@pytest.mark.parametrize("lic,ok", [("CC BY-SA 4.0", True), ("CC BY 2.0", True), ("CC0", True), ("Public domain", True),
                                    ("CC BY-NC 4.0", False), ("CC BY-ND 2.0", False), ("CC BY-NC-SA 3.0", False),
                                    ("GFDL", False), ("", False), ("All rights reserved", False)])
def test_license_filter(lic, ok):
    assert F.license_ok(lic) is ok


def test_candidate_filters():
    good = page(1, "File:Toyota Vios 2019.jpg")
    assert F.candidate_ok(good, "toyota", "vios")["author"] == "Nguyen An"            # bỏ HTML trong tên tác giả
    assert F.candidate_ok(page(1, "File:Toyota Vios.jpg", lic="CC BY-NC 4.0"), "toyota", "vios") is None
    assert F.candidate_ok(page(1, "File:Toyota Vios.jpg", w=500, h=300), "toyota", "vios") is None       # quá nhỏ
    assert F.candidate_ok(page(1, "File:Toyota Vios.jpg", w=1000, h=1500), "toyota", "vios") is None     # ảnh dọc
    assert F.candidate_ok(page(1, "File:Toyota Vios.gif", mime="image/gif"), "toyota", "vios") is None
    assert F.candidate_ok(page(1, "File:Toyota Vios interior.jpg"), "toyota", "vios") is None            # nội thất
    assert F.candidate_ok(page(1, "File:Honda City 2019.jpg", cats="Honda City"), "toyota", "vios") is None  # sai hãng/dòng
    assert F.candidate_ok({"title": "x", "imageinfo": []}, "toyota", "vios") is None


def test_pick_prefers_search_rank_among_valid():
    data = api(page(3, "File:Toyota Vios C.jpg"), page(1, "File:Toyota Vios interior.jpg"), page(2, "File:Toyota Vios B.jpg"))
    assert F.pick_candidate(data, "toyota", "vios")["title"] == "File:Toyota Vios B.jpg"      # rank 1 bị loại (nội thất) -> rank 2
    assert F.pick_candidate(api(), "toyota", "vios") is None and F.pick_candidate({}, "toyota", "vios") is None


def _args(tmp_path, **kw):
    base = dict(clean=str(tmp_path / "clean.csv"), out=str(tmp_path / "img"), top=10, min_listings=1, only=["toyota/vios", "honda/city", "kia/seltos"],
                delay=0, max_bytes=800_000, refresh=False, dry_run=False, drop=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _fake(calls):
    def s(brand, model):
        calls.append((brand, model))
        if (brand, model) == ("toyota", "vios"):
            return api(page(1, "File:Toyota Vios 2019.jpg"))
        if (brand, model) == ("kia", "seltos"):
            raise RuntimeError("mạng lỗi")
        return api()                              # honda/city: không có ảnh đạt lọc
    return s, (lambda url: b"\xff\xd8\xff\xe0JPEGDATA")


def test_run_end_to_end_resume_and_drop(tmp_path):
    calls = []
    s, fch = _fake(calls)
    st = F.run(_args(tmp_path), fetch_fn=fch, search_fn=s, sleep=lambda x: None)
    assert st == {"ok": 1, "skipped": 0, "not_found": 1, "error": 1}                      # lỗi một dòng không làm hỏng cả lượt
    out = tmp_path / "img"
    assert (out / "toyota" / "vios.jpg").read_bytes().startswith(b"\xff\xd8")
    rows = list(csv.DictReader(open(out / "CREDITS.csv", encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["author"] == "Nguyen An" and rows[0]["license"] == "CC BY-SA 4.0" and rows[0]["file"] == "toyota/vios.jpg"
    assert "Toyota Vios 2019" in (out / "review.html").read_text(encoding="utf-8") and "--drop toyota/vios" in (out / "review.html").read_text(encoding="utf-8")

    calls.clear()                                                                           # chạy lại: bỏ qua dòng đã có ảnh
    st = F.run(_args(tmp_path), fetch_fn=fch, search_fn=s, sleep=lambda x: None)
    assert st["skipped"] == 1 and ("toyota", "vios") not in calls

    # tích hợp: Recommender đọc đúng CREDITS.csv do script ghi
    import pandas as pd
    from recommend import Recommender
    rows = [dict(source="bonbanh", source_id=i, brand="toyota", model="vios", variant=None, body_type="sedan", year=2019, mileage_km=40000 + i,
                 engine_capacity=1.5, transmission="automatic", fuel_type="gasoline", drivetrain=None, seats=5, origin="domestic",
                 province=None, color=None, price=(480 + i) * 10**6, group_id=i) for i in range(3)]
    r = Recommender(pd.DataFrame(rows), images_dir=out)
    x = r.recommend({"brand": "toyota", "model": "vios"}, "rule_based", 1)["items"][0]
    assert x["image_url"].startswith("/car-images/toyota/vios.jpg?v=") and x["image_credit"]["license"] == "CC BY-SA 4.0"

    assert F.run(_args(tmp_path, drop="toyota/vios"))["dropped"] == 1                       # xóa ảnh chọn sai
    assert not (out / "toyota" / "vios.jpg").exists() and list(csv.DictReader(open(out / "CREDITS.csv", encoding="utf-8"))) == []


def test_dry_run_and_oversize(tmp_path):
    calls = []
    s, _ = _fake(calls)
    F.run(_args(tmp_path, dry_run=True), search_fn=s)
    assert calls == [] and not (tmp_path / "img" / "CREDITS.csv").exists()
    st = F.run(_args(tmp_path, only=["toyota/vios"], max_bytes=5), fetch_fn=lambda u: b"x" * 100, search_fn=s, sleep=lambda x: None)
    assert st["error"] == 1 and not (tmp_path / "img" / "toyota" / "vios.jpg").exists()      # ảnh quá lớn bị từ chối


def test_top_models_reads_clean_csv(tmp_path):
    import pandas as pd
    pd.DataFrame({"brand": ["Toyota"] * 5 + ["kia"] * 3 + ["honda"], "model": ["Vios"] * 5 + ["seltos"] * 3 + ["other"]}).to_csv(tmp_path / "c.csv", index=False)
    assert F.top_models(tmp_path / "c.csv", 5, 2) == [("toyota", "vios"), ("kia", "seltos")]
