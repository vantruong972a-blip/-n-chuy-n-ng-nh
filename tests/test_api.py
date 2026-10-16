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
    assert client.get("/api/meta").json()["unit"]
    m = client.get("/api/metrics").json()
    assert m["models"] and "RMSE" in m["models"][0]


def test_predict_valid(client, valid_input):
    r = client.post("/api/predict", json={"values": valid_input})
    assert r.status_code == 200
    j = r.json()
    assert j["price"] > 0 and j["unit"] and j["model_label"] and j["model_version"]


def test_predict_each_model_same_input(client, valid_input):
    names = list(client.get("/api/meta").json()["models"])
    prices = {n: client.post("/api/predict", json={"values": valid_input, "model": n}).json()["price"] for n in names}
    assert len(set(prices.values())) == len(names)


@pytest.mark.parametrize("patch", [{"enginesize": "abc"}, {"enginesize": 99999}, {"carbody": "boat"}, {"citympg": ""}])
def test_predict_invalid(client, valid_input, patch):
    assert client.post("/api/predict", json={"values": {**valid_input, **patch}}).status_code == 422


def test_unknown_model(client, valid_input):
    assert client.post("/api/predict", json={"values": valid_input, "model": "zzz"}).status_code == 422


def test_feedback_and_monitoring(client, valid_input):
    pid = client.post("/api/predict", json={"values": valid_input}).json()["prediction_id"]
    assert client.post("/api/feedback", json={"prediction_id": pid, "actual_price": 9000}).status_code == 200
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
