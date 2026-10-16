import numpy as np
import pandas as pd
import pytest

import config as C
from preprocess import build_preprocessor, get_split, load_clean


@pytest.mark.parametrize("ts", [0.4, 0.3, 0.2, 0.1])
def test_split_sizes_and_disjoint(ts):
    X_tr, X_te, y_tr, y_te = get_split(ts)
    n = len(X_tr) + len(X_te)
    assert abs(len(X_te) / n - ts) < 0.02
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert C.TARGET not in X_tr.columns and len(X_tr) == len(y_tr)


def test_split_reproducible():
    a = get_split(0.2)[0].index
    b = get_split(0.2)[0].index
    assert list(a) == list(b)


def test_id_columns_dropped():
    df = load_clean()
    assert not set(C.DROP_COLS) & set(df.columns)
    assert df[C.TARGET].notna().all()


def test_preprocessor_fit_on_train_only():
    X_tr, X_te, _, _ = get_split(0.2)
    pre = build_preprocessor(X_tr).fit(X_tr)
    A_tr, A_te = pre.transform(X_tr), pre.transform(X_te)
    n_num = len(pre.transformers_[0][2])
    assert np.allclose(A_tr[:, :n_num].mean(0), 0, atol=1e-8)       # chuẩn hóa theo train
    assert not np.allclose(A_te[:, :n_num].mean(0), 0, atol=1e-8)   # test không bị fit lại
    assert A_tr.shape[1] == A_te.shape[1]


def test_missing_target_raises(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"a": range(50)}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_clean(p)
