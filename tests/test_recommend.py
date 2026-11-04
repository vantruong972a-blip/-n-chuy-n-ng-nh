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
    r = rec.recommend({"brand": "toyota", "budget_min": 400e6, "budget_max": 600e6}, "content_based", 10)
    assert r["items"] and all(x["brand"] == "toyota" and 400e6 <= x["price"] <= 600e6 for x in r["items"])
    r = rec.recommend({"transmission": "manual"}, "content_based", 10)
    assert [x["model"] for x in r["items"]] == ["city"]


def test_no_match_returns_empty_not_fake(rec):
    assert rec.recommend({"brand": "ferrari"}, "knn", 5) == {"items": [], "method": "knn", "total": 0}
    assert rec.recommend({"budget_min": 1, "budget_max": 2}, "content_based", 5)["items"] == []


def test_reposted_car_appears_once(rec):
    r = rec.recommend({"brand": "toyota", "model": "vios", "budget_max": 400e6}, "content_based", 10)
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
    x = rec.recommend({"brand": "toyota", "model": "vios", "budget_max": 400e6}, "content_based", 1)["items"][0]
    assert x["image_url"] is None and x["price"] == 380 * 10**6 and x["year"] == 2019
    assert any("thấp hơn" in h for h in x["highlights"]) and len(x["highlights"]) <= 4
    y = rec.recommend({"brand": "honda"}, "content_based", 1)["items"][0]
    assert y["mileage_km"] is None and not any("Đã đi ít" in h for h in y["highlights"])
    z = rec.recommend({"brand": "toyota", "model": "camry"}, "content_based", 1)["items"][0]
    assert any("Đã đi ít" in h for h in z["highlights"]) and "Số tự động" not in z["highlights"]


def test_get_detail_and_similar(rec):
    cid = rec.recommend({"brand": "toyota", "model": "camry"}, "content_based", 1)["items"][0]["id"]
    d = rec.get(cid)
    assert d["id"] == cid and cid not in d["similar_ids"] and len(d["similar_ids"]) == 4
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
    monkeypatch.setattr(main, "load_recommender", lambda path: rec)
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

    def boom(path):
        raise DataNotAvailable("Chưa có dữ liệu")
    monkeypatch.setattr(main, "load_recommender", boom)
    r = TestClient(main.app).post("/api/recommend", json={})
    assert r.status_code == 503 and "dữ liệu" in r.json()["detail"]
