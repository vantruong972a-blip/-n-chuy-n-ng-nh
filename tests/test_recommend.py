import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recommend import DataNotAvailable, Recommender, load_recommender  # noqa: E402


def _df():
    rows = []
    rid = 0

    def add(brand, model, year, km, price, fuel="gasoline", trans="automatic", body="sedan", group=None, src="bonbanh"):
        nonlocal rid
        rid += 1
        rows.append(dict(source=src, source_id=rid, brand=brand, model=model, variant=None, body_type=body, year=year, mileage_km=km,
                         engine_capacity=1.5, transmission=trans, fuel_type=fuel, drivetrain=None, seats=5, origin="domestic",
                         province=None, color=None, price=price, group_id=group if group is not None else rid))
    for i in range(6):                      # 6 tin Vios 2019 -> đủ >=5 để có trung vị nhóm
        add("toyota", "vios", 2019, 40000 + i * 1000, (480 + i * 10) * 10**6)
    add("toyota", "vios", 2019, 50000, 380 * 10**6)           # rẻ hơn hẳn mặt bằng
    add("toyota", "vios", 2019, 50000, 380 * 10**6, group=rows[-1]["group_id"], src="chotot")   # cùng xe đăng lại
    add("toyota", "camry", 2021, 20000, 1_000 * 10**6)
    add("honda", "city", 2018, None, 450 * 10**6, trans="manual")
    add("ford", "ranger", 2020, 60000, 700 * 10**6, fuel="diesel", body="pickup")
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def rec():
    return Recommender(_df())


def test_content_based_filters_are_hard(rec):
    r = rec.recommend({"brand": "toyota", "budget_min": 400e6, "budget_max": 600e6}, "rule_based", 10)
    assert r["items"] and all(x["brand"] == "toyota" and 400e6 <= x["price"] <= 600e6 for x in r["items"])
    r = rec.recommend({"transmission": "manual"}, "rule_based", 10)
    assert [x["model"] for x in r["items"]] == ["city"]


def test_no_match_returns_empty_not_fake(rec):
    r = rec.recommend({"brand": "ferrari"}, "knn", 5)
    assert r["items"] == [] and r["total"] == 0 and r["method"] == "knn" and r["notes"]
    assert rec.recommend({"budget_min": 1, "budget_max": 2}, "rule_based", 5)["items"] == []


def test_reposted_car_appears_once(rec):
    r = rec.recommend({"brand": "toyota", "model": "vios", "budget_max": 400e6}, "rule_based", 10)
    assert len(r["items"]) == 1 and r["total"] == 1


def test_knn_returns_nearest_in_price(rec):
    r = rec.recommend({"budget_min": 400e6, "budget_max": 500e6}, "knn", 3)
    assert all(400e6 <= x["price"] <= 500e6 for x in r["items"])     # ngân sách là bộ lọc cứng
    assert abs(r["items"][0]["price"] - 450e6) <= 50e6                # xe gần nhất nằm sát giữa khoảng ngân sách
    assert len(r["items"]) == 3 and r["method"] == "knn"


def test_knn_soft_attributes_do_not_filter(rec):
    r = rec.recommend({"fuel_type": "diesel", "budget_min": 650e6, "budget_max": 750e6}, "knn", 3)
    assert r["items"][0]["model"] == "ranger"
    r = rec.recommend({"fuel_type": "electric"}, "knn", 3)           # không xe điện, vẫn trả xe gần nhất (tiêu chí mềm)
    assert r["items"]


def test_listing_fields_and_highlights(rec):
    x = rec.recommend({"brand": "toyota", "model": "vios", "budget_max": 400e6}, "rule_based", 1)["items"][0]
    assert x["image_url"] is None and x["price"] == 380 * 10**6 and x["year"] == 2019
    assert any("thấp hơn" in h for h in x["highlights"]) and len(x["highlights"]) <= 4
    y = rec.recommend({"brand": "honda"}, "rule_based", 1)["items"][0]
    assert y["mileage_km"] is None and not any("Đã đi ít" in h for h in y["highlights"])
    z = rec.recommend({"brand": "toyota", "model": "camry"}, "rule_based", 1)["items"][0]
    assert any("Đã đi ít" in h for h in z["highlights"]) and "Số tự động" not in z["highlights"]


def test_get_detail_and_similar(rec):
    cid = rec.recommend({"brand": "toyota", "model": "camry"}, "rule_based", 1)["items"][0]["id"]
    d = rec.get(cid)
    assert d["id"] == cid and cid not in d["similar_ids"] and len(d["similar_ids"]) == 4
    assert [x["id"] for x in d["similar"]] == d["similar_ids"] and all("similar" not in x for x in d["similar"])
    assert rec.get("khong-co") is None


def test_missing_data_file(tmp_path):
    with pytest.raises(DataNotAvailable):
        load_recommender(tmp_path / "khong_co.csv")


# ---- API ----
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(monkeypatch, rec):
    from app import main
    monkeypatch.setattr(main, "load_recommender", lambda *a, **k: rec)
    return TestClient(main.app)


def test_api_recommend_and_detail(client):
    r = client.post("/api/recommend", json={"brand": "toyota", "budget_min": 400e6, "budget_max": 600e6, "method": "knn", "limit": 3})
    assert r.status_code == 200 and 0 < len(r.json()["items"]) <= 3
    cid = r.json()["items"][0]["id"]
    d = client.get(f"/api/cars/{cid}")
    assert d.status_code == 200 and d.json()["id"] == cid and d.json()["similar_ids"]
    assert client.get("/api/cars/khong-co").status_code == 404


def test_api_recommend_validation(client):
    assert client.post("/api/recommend", json={"budget_min": 9, "budget_max": 1}).status_code == 422
    assert client.post("/api/recommend", json={"method": "abc"}).status_code == 422
    assert client.post("/api/recommend", json={"limit": 0}).status_code == 422
    assert client.post("/api/recommend", json={}).status_code == 200      # mọi bộ lọc đều tùy chọn


def test_api_503_when_no_data(monkeypatch):
    from app import main

    def boom(*a, **k):
        raise DataNotAvailable("Chưa có dữ liệu")
    monkeypatch.setattr(main, "load_recommender", boom)
    r = TestClient(main.app).post("/api/recommend", json={})
    assert r.status_code == 503 and "dữ liệu" in r.json()["detail"]


# ---- khớp tên mềm, khử trùng, ảnh minh họa ----
from recommend import norm_name, resolve_names  # noqa: E402


def test_norm_name():
    assert norm_name("Vios G") == "viosg" and norm_name("Mercedes-Benz") == "mercedesbenz" and norm_name("Hyundai Grand i10") == "hyundaigrandi10"
    assert norm_name("") == "" and norm_name(None) == "none"


def test_resolve_names_kinds():
    names = {"vios": 100, "viosa": 5, "camry": 50, "c": 30, "cx": 20}
    assert resolve_names("Vios", names)["kind"] == "exact"
    r = resolve_names("vios g", names)                 # người dùng gõ thêm: chứa tên có thật
    assert r["kind"] == "partial" and "vios" in r["matched"]
    assert resolve_names("camri", names)["kind"] == "fuzzy"
    r = resolve_names("khongco", names)
    assert r["kind"] == "none" and r["matched"] == []
    assert resolve_names("c", names)["kind"] == "exact"   # tên 1 ký tự chỉ khớp đúng, không khớp lan man
    assert resolve_names("", names)["kind"] == "none"


def test_soft_match_note_and_unknown_model(rec):
    r = rec.recommend({"brand": "Toyota", "model": "vios g"}, "rule_based", 5)
    assert r["items"] and all(x["model"] == "vios" for x in r["items"])
    assert any("gần đúng" in n for n in r["notes"])
    r = rec.recommend({"brand": "toyota", "model": "zzzzzz"}, "rule_based", 5)
    assert r["items"] == [] and r["total"] == 0 and r["notes"] and r["suggestion_for"] == "model"
    assert rec.recommend({"brand": "toyota", "model": "vios"}, "rule_based", 5)["notes"] == []   # khớp đúng: không ghi chú
    r = rec.recommend({"brand": "hang-la"}, "knn", 5)
    assert r["items"] == [] and r["suggestion_for"] == "brand"


def test_near_duplicates_with_different_group_are_removed():
    rows = []
    for i, g in enumerate([1, 2, 3]):          # 3 tin giống hệt về hãng/dòng/năm/giá/km nhưng khác group_id
        rows.append(dict(source="chotot", source_id=i, brand="vinfast", model="vf e34", variant=None, body_type="suv_crossover", year=2022,
                         mileage_km=None, engine_capacity=None, transmission="automatic", fuel_type="electric", drivetrain=None, seats=5,
                         origin="domestic", province=None, color="trắng", price=590_000_000, group_id=g))
    rows.append({**rows[0], "source_id": 9, "group_id": 9, "price": 600_000_000})
    r = Recommender(pd.DataFrame(rows))
    out = r.recommend({"brand": "vinfast"}, "rule_based", 10)
    assert len(out["items"]) == 2 and out["total"] == 2
    assert len(r.get(out["items"][0]["id"])["similar_ids"]) <= 1


def test_images_only_when_file_exists_and_credit_valid(tmp_path):
    (tmp_path / "toyota").mkdir()
    (tmp_path / "toyota" / "vios.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "CREDITS.csv").write_text(
        "brand,model,file,title,author,license,license_url,source_url\n"
        "toyota,vios,toyota/vios.jpg,File:Vios.jpg,Nguyen A,CC BY-SA 4.0,https://creativecommons.org/licenses/by-sa/4.0/,https://commons.wikimedia.org/wiki/File:Vios.jpg\n"
        "toyota,camry,toyota/camry.jpg,File:Camry.jpg,B,CC0,,\n"              # file không tồn tại
        "toyota,city,../secret.jpg,File:X.jpg,C,CC0,,\n"                        # cố thoát thư mục
        "toyota,yaris,toyota/yaris.txt,File:Y.jpg,D,CC0,,\n", encoding="utf-8")   # đuôi không hợp lệ
    r = Recommender(_df(), images_dir=tmp_path)
    vios = r.recommend({"brand": "toyota", "model": "vios"}, "rule_based", 1)["items"][0]
    assert vios["image_url"].startswith("/car-images/toyota/vios.jpg?v=") and vios["image_kind"] == "model_illustration"
    assert vios["image_credit"]["author"] == "Nguyen A" and vios["image_credit"]["license"] == "CC BY-SA 4.0"
    camry = r.recommend({"brand": "toyota", "model": "camry"}, "rule_based", 1)["items"][0]
    assert camry["image_url"] is None and camry["image_credit"] is None
    assert len(r.images) == 1


def test_api_serves_images_and_blocks_traversal(tmp_path, monkeypatch):
    from app import main
    (tmp_path / "toyota").mkdir()
    (tmp_path / "toyota" / "vios.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    monkeypatch.setattr(main, "IMAGES_DIR", tmp_path)
    from starlette.staticfiles import StaticFiles
    main.app.router.routes.insert(0, __import__("starlette.routing", fromlist=["Mount"]).Mount("/car-images", app=StaticFiles(directory=tmp_path), name="t-img"))
    try:
        c = TestClient(main.app)
        assert c.get("/car-images/toyota/vios.jpg").status_code == 200
        assert c.get("/car-images/..%2f..%2fetc%2fpasswd").status_code in (400, 404)
        assert c.get("/car-images/khong-co.jpg").status_code == 404
    finally:
        main.app.router.routes.pop(0)


def test_missing_attributes_do_not_collapse_signatures():
    """Hồi quy: tin thiếu nhiên liệu/hộp số/km không được bị coi là trùng nhau chỉ vì cùng thiếu."""
    base = dict(source="bonbanh", variant=None, body_type=None, engine_capacity=None, transmission=None, fuel_type=None, drivetrain=None,
                seats=None, origin=None, province=None, color=None, mileage_km=None)
    rows = [dict(base, source_id=i, brand="kia", model=m, year=2015 + i % 3, price=(300 + 10 * i) * 10**6, group_id=i)
            for i, m in enumerate(["morning", "cerato", "seltos", "carens", "rio"])]
    r = Recommender(pd.DataFrame(rows))
    assert r.df["_sig"].notna().all() and r.df["_sig"].nunique() == 5
    assert r.recommend({}, "rule_based", 10)["total"] == 5


# ---------- Content-Based Filtering đúng nghĩa ----------
def _cbf_df():
    rows = []
    def add(i, brand, model, year, price, fuel, trans, body, km=50000, gid=None):
        rows.append(dict(source="bonbanh", source_id=str(i), brand=brand, model=model, year=year, price=price, fuel_type=fuel,
                         transmission=trans, body_type=body, origin="domestic", mileage_km=km, engine_capacity=1.5, seats=5,
                         province="hanoi", group_id=gid or f"g{i}"))
    for i in range(12):
        add(i, "toyota", "vios", 2016 + i % 4, 380e6 + i * 5e6, "gasoline", "automatic", "sedan", 40000 + i * 3000)
    for i in range(12, 24):
        add(i, "honda", "city", 2016 + i % 4, 390e6 + i * 5e6, "gasoline", "manual", "sedan", 50000 + i * 1000)
    for i in range(24, 36):
        add(i, "vinfast", "vf e34", 2022, 600e6 + i * 1e6, "electric", "automatic", "suv_crossover", 10000 + i * 100)
    return pd.DataFrame(rows)


def test_cbf_profile_from_soft_attributes_and_scores():
    rec = Recommender(_cbf_df())
    r = rec.recommend({"fuel_type": "electric"}, "content_based", 5)
    assert r["items"] and all(x["fuel_type"] == "electric" for x in r["items"])     # thuộc tính mềm nhưng khớp cao nhất đứng đầu
    assert all(0 <= x["match_score"] <= 1 for x in r["items"])
    s = [x["match_score"] for x in r["items"]]
    assert s == sorted(s, reverse=True)
    assert rec.recommend({"transmission": "manual"}, "content_based", 3)["items"][0]["brand"] == "honda"


def test_cbf_soft_attribute_without_match_still_returns_nearest():
    rec = Recommender(_cbf_df())
    r = rec.recommend({"body_type": "pickup", "budget_min": 380e6, "budget_max": 420e6}, "content_based", 3)
    assert r["items"] and all(380e6 <= x["price"] <= 420e6 for x in r["items"])     # ngân sách luôn là lọc cứng


def test_cbf_liked_items_profile_and_exclusion():
    rec = Recommender(_cbf_df())
    liked = "bonbanh-30"
    r = rec.recommend({}, "content_based", 6, liked_ids=[liked])
    assert r["items"] and liked not in [x["id"] for x in r["items"]]
    assert r["items"][0]["brand"] == "vinfast"                                       # xe giống xe đã thích nhất
    assert r["items"][0]["match_score"] > r["items"][-1]["match_score"] - 1e-9
    # xe thích khác loại kéo kết quả sang loại đó
    assert rec.recommend({}, "content_based", 3, liked_ids=["bonbanh-1"])["items"][0]["brand"] == "toyota"


def test_cbf_unknown_liked_id_noted_and_no_profile_fallback():
    rec = Recommender(_cbf_df())
    r = rec.recommend({}, "content_based", 3, liked_ids=["khong-co"])
    assert any("khong-co" in n for n in r["notes"]) and r["items"]                  # không có hồ sơ: xếp theo luật, có ghi chú
    assert "match_score" not in r["items"][0]


def test_cbf_tfidf_rare_model_weighs_more_than_common_brand_fuel():
    rec = Recommender(_cbf_df())
    col_f, w_f = rec._cb_token[("fuel_type", "gasoline")]
    col_e, w_e = rec._cb_token[("fuel_type", "electric")]
    assert w_e > w_f                                                                # giá trị hiếm có idf lớn hơn


def test_rule_based_still_hard_filters():
    rec = Recommender(_cbf_df())
    r = rec.recommend({"transmission": "manual"}, "rule_based", 20)
    assert r["items"] and all(x["transmission"] == "manual" for x in r["items"])


def test_unknown_model_other_is_not_recommended():
    """Dòng xe 'other' gộp nhiều mẫu không rõ nên không được đưa vào gợi ý."""
    import pandas as pd
    from src.recommend import Recommender

    rows = []
    for i, m in enumerate(["vios", "other", "vios", "khac"]):
        rows.append({"source": "t", "source_id": str(i), "brand": "toyota", "model": m, "year": 2018 + i % 2,
                     "price": 400e6 + i * 1e6, "group_id": i, "mileage_km": 30000 + i,
                     "fuel_type": "petrol", "transmission": "at", "body_type": "sedan", "origin": "domestic",
                     "engine_capacity": 1.5, "seats": 5, "color": "x", "variant": "x", "province": "ha-noi", "drivetrain": "fwd"})
    r = Recommender(pd.DataFrame(rows))
    assert set(r.df["model"]) == {"vios"}
    out = r.recommend({"brand": "toyota"}, "rule_based", limit=10)
    assert all(x["model"] == "vios" for x in out["items"])
