import pytest

from inference import InvalidInput, ModelNotAvailable, Predictor


def test_valid_prediction(trained, valid_input):
    p = Predictor()
    r = p.predict(valid_input)
    assert r["price"] > 0 and r["unit"] and r["model_version"] and r["model_label"]
    assert len(r["all"]) == len(p.models)


def test_column_order_does_not_matter(trained, valid_input):
    p = Predictor()
    a = p.predict(valid_input)["price"]
    b = p.predict(dict(reversed(list(valid_input.items()))))["price"]
    assert a == pytest.approx(b)


def test_each_model_selectable(trained, valid_input):
    p = Predictor()
    for m in p.models:
        assert p.predict(valid_input, m)["model"] == m


@pytest.mark.parametrize("patch", [{"enginesize": "abc"}, {"enginesize": 99999}, {"carbody": "boat"},
                                   {"citympg": None}, {"horsepower": ""}, {"unknown_col": 1}])
def test_invalid_inputs(trained, valid_input, patch):
    with pytest.raises(InvalidInput):
        Predictor().predict({**valid_input, **patch})


def test_unknown_model(trained, valid_input):
    with pytest.raises(InvalidInput):
        Predictor().predict(valid_input, "no_such_model")


def test_missing_models_raises(monkeypatch, tmp_path):
    import config as C
    monkeypatch.setattr(C, "MODELS", tmp_path)
    with pytest.raises(ModelNotAvailable):
        Predictor()
