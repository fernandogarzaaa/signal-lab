"""Tests for the TAILQ challenger and naive arm (synthetic, no network).

Covers: LightGBM quantile fits and predicts finite 3-day quantiles,
the frozen hyperparameters are exactly as pre-registered, the fit is
deterministic given seed 7, forecasts beat a constant bad forecast on
heteroskedastic synthetic data, and naive_3d_quantile math.
"""

import numpy as np
import pytest

from signal_lab.tailq import models as models_mod


def _synthetic_Xy(n=600, p=15, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, p))
    # Heteroskedastic: tail scale grows with |X0|; true 5% quantile is
    # a nonlinear function of X0.
    scale = 0.01 * (1.0 + 2.0 * np.abs(X[:, 0]))
    y = rng.normal(0, 1, n) * scale - 0.005 * X[:, 1]
    return X, y


def test_frozen_hyperparameters():
    p = models_mod.LGBM_QUANTILE_PARAMS
    assert p["objective"] == "quantile"
    assert p["alpha"] == 0.05
    assert p["n_estimators"] == 300
    assert p["learning_rate"] == 0.05
    assert p["num_leaves"] == 15
    assert p["min_child_samples"] == 40
    assert p["feature_fraction"] == 0.8
    assert p["bagging_fraction"] == 0.8
    assert p["bagging_freq"] == 1
    assert p["lambda_l2"] == 1.0
    assert p["seed"] == 7
    assert p["deterministic"] is True


def test_fit_predict_finite():
    X, y = _synthetic_Xy()
    booster = models_mod.fit_lgbm_quantile(X, y)
    q = models_mod.predict_lgbm_quantile(booster, X)
    assert q.shape == (len(y),)
    assert np.all(np.isfinite(q))


def test_fit_is_deterministic():
    X, y = _synthetic_Xy()
    b1 = models_mod.fit_lgbm_quantile(X, y)
    b2 = models_mod.fit_lgbm_quantile(X, y)
    q1 = models_mod.predict_lgbm_quantile(b1, X)
    q2 = models_mod.predict_lgbm_quantile(b2, X)
    np.testing.assert_array_equal(q1, q2)


def test_beats_constant_bad_forecast():
    from signal_lab.tailq import metrics as met
    X, y = _synthetic_Xy(n=1000)
    booster = models_mod.fit_lgbm_quantile(X[:700], y[:700])
    q = models_mod.predict_lgbm_quantile(booster, X[700:])
    bad = np.full_like(q, 0.0)  # forecasting the median-ish zero
    assert met.mean_pinball(y[700:], q) < met.mean_pinball(y[700:], bad)


def test_rejects_bad_inputs():
    X, y = _synthetic_Xy()
    with pytest.raises(ValueError):
        models_mod.fit_lgbm_quantile(np.empty((0, 15)), np.empty(0))
    Xb = X.copy()
    Xb[0, 0] = np.nan
    with pytest.raises(ValueError):
        models_mod.fit_lgbm_quantile(Xb, y)
    booster = models_mod.fit_lgbm_quantile(X, y)
    with pytest.raises(ValueError):
        models_mod.predict_lgbm_quantile(booster, np.empty((0, 15)))


def test_naive_3d_quantile_math():
    rng = np.random.default_rng(7)
    r = rng.normal(-0.001, 0.02, 300)
    q = models_mod.naive_3d_quantile(r, window=252, tau=0.05)
    assert q == pytest.approx(float(np.quantile(r[-252:], 0.05)) * np.sqrt(3.0))
    assert q < 0


def test_naive_rejects_empty():
    with pytest.raises(ValueError):
        models_mod.naive_3d_quantile(np.array([np.nan, np.nan]))
