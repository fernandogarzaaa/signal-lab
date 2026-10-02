"""Tests for the HARVIX ridge arms: frozen estimator behavior on 31/34 columns."""

import numpy as np
import pandas as pd
import pytest

from signal_lab.harvix import features as feat_mod
from signal_lab.vol import models as models_mod


def _design(n=400, seed=7):
    rng = np.random.default_rng(seed)
    cols = feat_mod.HARVIX_COLUMNS
    X = pd.DataFrame(rng.normal(0, 1, (n, len(cols))), columns=cols)
    # A vol-like target: positive, heteroskedastic-ish.
    y = pd.Series(np.abs(0.01 + 0.002 * X["rv_1d"] + 0.001 * X["vix_level"])
                  + rng.normal(0, 0.0005, n))
    return X, y


def test_ridge_harvix_fit_predict_finite_positive():
    X, y = _design()
    fitted = models_mod.fit_ridge(X[feat_mod.HARVIX_COLUMNS], y)
    fc = models_mod.predict_ridge(fitted, X[feat_mod.HARVIX_COLUMNS].iloc[300:])
    assert np.all(np.isfinite(fc)) and (fc > 0).all()
    assert (fc >= models_mod.FORECAST_FLOOR).all()


def test_ridge_har_arm_uses_31_columns():
    X, y = _design()
    fitted = models_mod.fit_ridge(X[feat_mod.HAR_COLUMNS], y)
    fc = models_mod.predict_ridge(fitted, X[feat_mod.HAR_COLUMNS].iloc[300:])
    assert np.all(np.isfinite(fc)) and (fc > 0).all()


def test_ridge_alpha_frozen():
    assert models_mod.RIDGE_ALPHA == 1.0


def test_ridge_deterministic():
    X, y = _design()
    f1 = models_mod.fit_ridge(X[feat_mod.HARVIX_COLUMNS], y)
    f2 = models_mod.fit_ridge(X[feat_mod.HARVIX_COLUMNS], y)
    Xt = X[feat_mod.HARVIX_COLUMNS].iloc[300:]
    np.testing.assert_array_equal(models_mod.predict_ridge(f1, Xt),
                                  models_mod.predict_ridge(f2, Xt))


def test_ridge_refuses_nonfinite():
    X, y = _design()
    Xb = X[feat_mod.HARVIX_COLUMNS].copy()
    Xb.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        models_mod.fit_ridge(Xb, y)


def test_vix_columns_change_the_fit():
    """Sanity: adding the 3 VIX columns actually moves the arm's forecasts."""
    X, y = _design()
    fh = models_mod.fit_ridge(X[feat_mod.HAR_COLUMNS], y)
    fx = models_mod.fit_ridge(X[feat_mod.HARVIX_COLUMNS], y)
    Xt = X.iloc[300:]
    a = models_mod.predict_ridge(fh, Xt[feat_mod.HAR_COLUMNS])
    b = models_mod.predict_ridge(fx, Xt[feat_mod.HARVIX_COLUMNS])
    assert not np.allclose(a, b)
