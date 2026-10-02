"""TAILQ challenger: LightGBM quantile regression + naive sanity arm.

Challenger spec (frozen in docs/TAILQ_PREREGISTRATION.md): LightGBM
with objective="quantile", alpha=0.05, on the 15 frozen point-in-time
features, refit per walk-forward fold. Hyperparameters are declared
below before any data is touched; no selection, no tuning.

Naive arm (sanity, no gate authority): unconditional empirical 5%
quantile of the trailing 252 trading-day log returns through t0,
times sqrt(3) (the frozen horizon scaling).
"""

from __future__ import annotations

import math

import numpy as np

TAU = 0.05
HORIZON_SCALE = math.sqrt(3.0)
NAIVE_WINDOW = 252

# Frozen challenger hyperparameters (pre-registered; not tuned).
LGBM_QUANTILE_PARAMS: dict = {
    "objective": "quantile",
    "alpha": TAU,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 7,
    "deterministic": True,
    "verbose": -1,
}


def fit_lgbm_quantile(X: np.ndarray, y: np.ndarray):
    """Fit the frozen LightGBM quantile challenger. Returns a Booster."""
    import lightgbm as lgb

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.shape[0] == 0:
        raise ValueError("fit_lgbm_quantile: empty training set")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"fit_lgbm_quantile: shape mismatch {X.shape} vs {y.shape}")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        raise ValueError(
            "fit_lgbm_quantile: NaN or inf in training data; refusing to fit")
    train = lgb.Dataset(X, label=y)
    booster = lgb.train(dict(LGBM_QUANTILE_PARAMS), train)
    return booster


def predict_lgbm_quantile(booster, X: np.ndarray) -> np.ndarray:
    """3-day 5% quantile forecasts from a fitted challenger."""
    import lightgbm as lgb  # noqa: F401 (documents the booster type)

    X = np.asarray(X, dtype=float)
    if X.shape[0] == 0:
        raise ValueError("predict_lgbm_quantile: empty input")
    if not np.all(np.isfinite(X)):
        raise ValueError(
            "predict_lgbm_quantile: NaN or inf in inputs; refusing to predict")
    q = np.asarray(booster.predict(X), dtype=float)
    if not np.all(np.isfinite(q)):
        raise ValueError("predict_lgbm_quantile: non-finite forecasts")
    return q


def naive_3d_quantile(returns: np.ndarray,
                      window: int = NAIVE_WINDOW,
                      tau: float = TAU) -> float:
    """Unconditional empirical tau-quantile of trailing returns, x sqrt(3)."""
    r = np.asarray(returns, dtype=float).ravel()
    rw = r[-window:] if r.size >= window else r
    rw = rw[np.isfinite(rw)]
    if rw.size < 2:
        raise ValueError("naive_3d_quantile: fewer than 2 finite returns")
    q = float(np.quantile(rw, tau))
    if not np.isfinite(q):
        raise ValueError("naive_3d_quantile: non-finite quantile")
    return q * HORIZON_SCALE
