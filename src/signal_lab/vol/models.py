"""Price ML arms: ridge regression and LightGBM on HAR-style features.

PRE-DECLARED hyperparameters (frozen before the selection run; see
docs/VOL_PREREGISTRATION.md "Selection honesty"). Modest by design: no
hyperparameter search happens inside this campaign.

- ridge: alpha = 1.0, features standardized (StandardScaler fit on train
  only, applied to validation/test: point-in-time safe).
- lightgbm: n_estimators = 300, learning_rate = 0.05, num_leaves = 31,
  min_child_samples = 100, feature_fraction = 0.8, bagging_fraction = 0.8,
  bagging_freq = 1, lambda_l2 = 1.0, random_state = 7.

Both arms predict realized_vol_5d (decimal daily). Forecasts are floored
at 1e-4 (1bp of daily vol): QLIKE needs strictly positive variance
forecasts, and the floor is economically negligible, never binding on
real data. The floor is an interface guard, not a tuned parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Pre-declared, frozen before the selection run. Do not tune.
RIDGE_ALPHA = 1.0
LGBM_PARAMS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "random_state": 7,
    "verbose": -1,
}

FORECAST_FLOOR = 1e-4

CHAMPION_CHOICES = ("ridge", "lightgbm")


@dataclass
class RidgeBundle:
    scaler: StandardScaler
    model: Ridge


def _check_xy(X: pd.DataFrame, y: pd.Series) -> None:
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
    if len(X) == 0:
        raise ValueError("empty training set")
    if not np.all(np.isfinite(X.to_numpy(dtype=float))):
        raise ValueError("non-finite values in feature matrix")
    if not np.all(np.isfinite(y.to_numpy(dtype=float))):
        raise ValueError("non-finite values in target")


def _floor(forecasts: np.ndarray) -> np.ndarray:
    fc = np.asarray(forecasts, dtype=float)
    if not np.all(np.isfinite(fc)):
        raise ValueError("non-finite model forecasts; refusing to score")
    return np.maximum(fc, FORECAST_FLOOR)


def fit_ridge(X_train: pd.DataFrame, y_train: pd.Series) -> RidgeBundle:
    """Fit the ridge arm (alpha = 1.0, standardized features)."""
    _check_xy(X_train, y_train)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(Xs, y_train.to_numpy(dtype=float))
    return RidgeBundle(scaler=scaler, model=model)


def predict_ridge(bundle: RidgeBundle, X: pd.DataFrame) -> np.ndarray:
    """Predict realized vol with the ridge arm (floored at 1e-4)."""
    Xs = bundle.scaler.transform(X)
    return _floor(bundle.model.predict(Xs))


def fit_lightgbm(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMRegressor:
    """Fit the LightGBM arm with pre-declared hyperparameters."""
    _check_xy(X_train, y_train)
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train.to_numpy(dtype=float))
    return model


def predict_lightgbm(model: lgb.LGBMRegressor, X: pd.DataFrame) -> np.ndarray:
    """Predict realized vol with the LightGBM arm (floored at 1e-4)."""
    return _floor(model.predict(X))


def fit_arm(name: str, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit a named arm ('ridge' | 'lightgbm')."""
    if name == "ridge":
        return fit_ridge(X_train, y_train)
    if name == "lightgbm":
        return fit_lightgbm(X_train, y_train)
    raise ValueError(f"unknown arm {name!r}; want one of {CHAMPION_CHOICES}")


def predict_arm(name: str, fitted, X: pd.DataFrame) -> np.ndarray:
    """Predict with a fitted named arm."""
    if name == "ridge":
        return predict_ridge(fitted, X)
    if name == "lightgbm":
        return predict_lightgbm(fitted, X)
    raise ValueError(f"unknown arm {name!r}; want one of {CHAMPION_CHOICES}")
