"""Tests for the EVENTVOL Model Confidence Set (synthetic, no network).

Checks: a clearly-worse model is eliminated, identical models all
survive, the procedure is deterministic given the seed, and bad input
fails loud.
"""

import numpy as np
import pandas as pd
import pytest

from signal_lab.eventvol import mcs as mcs_mod

N = 600
DAYS = pd.bdate_range("2023-01-02", periods=120)


def _t0(n):
    return pd.Series([DAYS[i % len(DAYS)] for i in range(n)])


def test_worse_model_eliminated():
    rng = np.random.default_rng(7)
    base = np.abs(rng.normal(0.5, 0.1, N))
    loss = pd.DataFrame({
        "good": base,
        "bad": base + 0.5,  # uniformly worse by a large margin
        "mid": base + rng.normal(0, 0.02, N),
    })
    res = mcs_mod.mcs(loss, _t0(N), n_boot=300, log=lambda *a: None)
    assert "bad" not in res["survivors"]
    assert "good" in res["survivors"]
    assert res["eliminated"][0][0] == "bad"


def test_identical_models_all_survive():
    rng = np.random.default_rng(7)
    base = np.abs(rng.normal(0.5, 0.1, N))
    loss = pd.DataFrame({"a": base, "b": base.copy(), "c": base.copy()})
    res = mcs_mod.mcs(loss, _t0(N), n_boot=300, log=lambda *a: None)
    assert set(res["survivors"]) == {"a", "b", "c"}
    assert res["eliminated"] == []


def test_deterministic_given_seed():
    rng = np.random.default_rng(7)
    base = np.abs(rng.normal(0.5, 0.1, N))
    loss = pd.DataFrame({
        "x": base,
        "y": base + rng.normal(0.05, 0.05, N),
        "z": base + 0.3,
    })
    r1 = mcs_mod.mcs(loss, _t0(N), n_boot=300, log=lambda *a: None)
    r2 = mcs_mod.mcs(loss, _t0(N), n_boot=300, log=lambda *a: None)
    assert r1["survivors"] == r2["survivors"]
    assert r1["eliminated"] == r2["eliminated"]


def test_mcs_p_values_sane():
    rng = np.random.default_rng(7)
    base = np.abs(rng.normal(0.5, 0.1, N))
    loss = pd.DataFrame({"x": base, "y": base + 0.4})
    res = mcs_mod.mcs(loss, _t0(N), n_boot=300, log=lambda *a: None)
    assert res["survivors"] == ["x"]
    for p in res["mcs_p"].values():
        assert 0.0 <= p <= 1.0


def test_nan_losses_raise():
    loss = pd.DataFrame({"a": [0.5, np.nan], "b": [0.5, 0.6]})
    with pytest.raises(ValueError, match="NaN"):
        mcs_mod.mcs(loss, _t0(2), log=lambda *a: None)


def test_single_model_raises():
    loss = pd.DataFrame({"a": [0.5, 0.6]})
    with pytest.raises(ValueError, match=">= 2 models"):
        mcs_mod.mcs(loss, _t0(2), log=lambda *a: None)
