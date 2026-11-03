import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(trained):
    from app import main
    main._state["predictor"] = None
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_meta_and_metrics(client):
    meta = client.get("/api/meta").json()
    assert meta["unit"] == "VND" and meta["required"] == ["brand", "model", "year"]
    assert {f["name"] for f in meta["features"]} == set(__import__("config").INPUT_FEATURES)
    m = client.get("/api/metrics").json()
    assert m["models"] and {"RMSE", "MAE", "R2", "medAPE"} <= set(m["models"][0]) and m["unit"] == "VND"


def test_predict_valid(client, valid_input):
    r = client.post("/api/predict", json={"values": valid_input})
    assert r.status_code == 200
    j = r.json()
    assert j["price"] > 0 and j["unit"] and j["model_label"] and j["model_version"]


def test_predict_each_model_same_input(client, valid_input):
    names = list(client.get("/api/meta").json()["models"])
    prices = {n: client.post("/api/predict", json={"values": valid_input, "model": n}).json()["price"] for n in names}
    assert len(set(prices.values())) == len(names)


@pytest.mark.parametrize("patch", [{"year": "abc"}, {"year": 1850}, {"year": 2040}, {"year": 2018.5}, {"mileage_km": -5},
                                   {"mileage_km": 99_999_999}, {"engine_capacity": 50}, {"seats": 99}, {"brand": "hãng-không-có"},
                                   {"transmission": "boat"}, {"year": True}, {"brand": ""}, {"model": None}, {"colour": "red"}])
def test_predict_invalid(client, valid_input, patch):
    r = client.post("/api/predict", json={"values": {**valid_input, **patch}})
    assert r.status_code == 422, patch
    assert isinstance(r.json()["detail"], list)


@pytest.mark.parametrize("missing", ["brand", "model", "year"])
def test_missing_required_field(client, valid_input, missing):
    v = {k: x for k, x in valid_input.items() if k != missing}
    r = client.post("/api/predict", json={"values": v})
    assert r.status_code == 422 and any(missing in e for e in r.json()["detail"])


def test_optional_fields_can_be_omitted_and_are_reported(client):
    r = client.post("/api/predict", json={"values": {"brand": "toyota", "model": "vios", "year": 2018}})
    assert r.status_code == 200
    j = r.json()
    assert j["price"] > 0 and "mileage_km" in j["defaults_used"] and "inputs" not in j


def test_input_is_normalised_and_warns(client):
    r = client.post("/api/predict", json={"values": {"brand": " Toyota ", "model": "VIOS", "year": "2018", "mileage_km": 0}})
    assert r.status_code == 200 and any("mileage_km=0" in w for w in r.json()["warnings"])
    r = client.post("/api/predict", json={"values": {"brand": "toyota", "model": "dòng-xe-lạ", "year": 2018}})
    assert r.status_code == 200 and r.json()["warnings"]


def test_older_car_is_not_priced_above_newer_same_car(client):
    """Kiểm tra hợp lý (sanity): cùng dòng xe, xe mới hơn không được rẻ hơn nhiều so với xe 15 năm tuổi."""
    p = lambda y: client.post("/api/predict", json={"values": {"brand": "toyota", "model": "vios", "year": y, "mileage_km": 50000}}).json()["price"]  # noqa: E731
    assert p(2021) > p(2007)


def test_unknown_model(client, valid_input):
    assert client.post("/api/predict", json={"values": valid_input, "model": "zzz"}).status_code == 422


def test_feedback_and_monitoring(client, valid_input):
    pid = client.post("/api/predict", json={"values": valid_input}).json()["prediction_id"]
    assert client.post("/api/feedback", json={"prediction_id": pid, "actual_price": 450_000_000}).status_code == 200
    assert client.post("/api/feedback", json={"prediction_id": "nope", "actual_price": 1}).status_code == 404
    mon = client.get("/api/monitoring").json()
    assert mon["n_predictions"] >= 1 and mon["feedback"]["n"] >= 1


def test_model_missing_returns_503(monkeypatch, tmp_path):
    import config as C
    from app import main
    monkeypatch.setattr(C, "MODELS", tmp_path)
    main._state["predictor"] = None
    c = TestClient(main.app)
    assert c.get("/health").status_code == 503
    assert c.post("/api/predict", json={"values": {}}).status_code == 503


def test_model_info(client):
    r = client.get("/api/model-info")
    assert r.status_code == 200
    j = r.json()
    assert j["model_version"] and j["data_version"] and j["n_train"] > 0 and j["n_test"] > 0
    assert j["serving_models"] and set(j["serving_models"]) <= set(j["models"])


def test_spa_fallback_and_legacy_ui(client, tmp_path, monkeypatch):
    from app import main
    (tmp_path / "index.html").write_text("<html>SPA-OK</html>", encoding="utf-8")
    (tmp_path / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    monkeypatch.setattr(main, "FRONTEND_DIST", tmp_path)
    assert "SPA-OK" in client.get("/").text
    assert "SPA-OK" in client.get("/dinh-gia").text            # route phía client
    assert client.get("/favicon.svg").status_code == 200
    assert client.get("/api/khong-ton-tai").status_code == 404   # API không bị nuốt bởi fallback
    assert client.get("/assets/khong-co.js").status_code == 404
    assert client.get("/../etc/passwd").status_code in (200, 404)
    assert "SPA-OK" not in client.get("/cu").text                # giao diện cũ vẫn truy cập được


def test_spa_without_build_falls_back_to_legacy(client, tmp_path, monkeypatch):
    from app import main
    monkeypatch.setattr(main, "FRONTEND_DIST", tmp_path / "chua-build")
    assert client.get("/").status_code == 200
    assert client.get("/trang-bat-ky").status_code == 404
