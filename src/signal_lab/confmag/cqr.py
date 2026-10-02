"""Conformalized Quantile Regression (CQR) for CONFMAG.

Standard split-conformal with absolute residuals yields constant-width
intervals, which would make width-based selection degenerate. CQR
(Romano, Patterson, Candes 2019) fits conditional quantiles and
conformalizes them, yielding adaptive interval widths that the
selection rule can rank.

Construction (docs/CONFMAG_PREREGISTRATION.md, frozen):
- Per fold, the train block is split TEMPORALLY: the most recent
  CAL_FRAC of train rows (by t0) form the calibration block; the rest
  is the proper-train block.
- Fit LightGBM quantile regressors on proper-train at TAU_LO=0.1 and
  TAU_HI=0.9 (frozen hyperparameters: the VOL LGBM set plus
  objective='quantile', alpha=tau).
- Conformity scores on calibration:
  E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)).
- q_hat = conformal_quantile(E, ALPHA) from
  signal_lab.models.conformal (the finite-sample corrected
  ceil((n+1)(1-alpha))/n order statistic).
- Test intervals: [q_lo(x) - q_hat, q_hi(x) + q_hat].
- Widths (conformalized): (q_hi - q_lo) + 2 * q_hat.

The point forecast (a separate LightGBM L2 regressor trained on the
fold's FULL train block) is scored only on the selected subset; the
quantile models serve the intervals/widths only.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from signal_lab.models.conformal import conformal_quantile
from signal_lab.vol.models import LGBM_PARAMS

TAU_LO = 0.1
TAU_HI = 0.9
ALPHA = 0.2
CAL_FRAC = 0.2
# Selection: test rows with width <= this percentile of calibration widths.
SELECT_Q = 0.25


def _quantile_params(tau: float) -> dict:
    p = dict(LGBM_PARAMS)
    p["objective"] = "quantile"
    p["alpha"] = tau
    return p


def fit_quantile_models(X_proper: pd.DataFrame, y_proper: pd.Series
                        ) -> tuple[lgb.LGBMRegressor, lgb.LGBMRegressor]:
    """Fit the frozen LightGBM quantile pair on the proper-train block."""
    y = y_proper.to_numpy(dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("[confmag] non-finite y in proper-train")
    m_lo = lgb.LGBMRegressor(**_quantile_params(TAU_LO))
    m_lo.fit(X_proper, y)
    m_hi = lgb.LGBMRegressor(**_quantile_params(TAU_HI))
    m_hi.fit(X_proper, y)
    return m_lo, m_hi


def fit_point_model(X_train: pd.DataFrame, y_train: pd.Series
                    ) -> lgb.LGBMRegressor:
    """Fit the frozen LightGBM L2 point regressor on the full train block."""
    y = y_train.to_numpy(dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("[confmag] non-finite y in train")
    m = lgb.LGBMRegressor(**LGBM_PARAMS)
    m.fit(X_train, y)
    return m


def predict_point(model: lgb.LGBMRegressor, X: pd.DataFrame) -> np.ndarray:
    """Point forecasts, floored at 0 (magnitudes are non-negative)."""
    return np.maximum(model.predict(X), 0.0)


def temporal_proper_cal(train_t0: pd.Series, cal_frac: float = CAL_FRAC
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Temporal proper-train / calibration split of a train block.

    ``train_t0``: the train block's t0 values (order = row order).
    Returns (proper_pos, cal_pos): positional index arrays. The most
    recent ``cal_frac`` rows by t0 form the calibration block.
    """
    t = pd.to_datetime(train_t0).to_numpy()
    order = np.argsort(t, kind="stable")
    n_cal = max(1, int(len(order) * cal_frac))
    cal_pos = order[-n_cal:]
    proper_pos = order[:-n_cal]
    if len(proper_pos) == 0:
        raise ValueError("[confmag] empty proper-train block")
    return proper_pos, cal_pos


def conformalize(m_lo: lgb.LGBMRegressor, m_hi: lgb.LGBMRegressor,
                 X_cal: pd.DataFrame, y_cal: pd.Series,
                 alpha: float = ALPHA) -> dict:
    """Fit the CQR correction on the calibration block.

    Returns q_hat, the calibration interval widths, and the selection
    threshold (SELECT_Q percentile of calibration widths).
    """
    y = y_cal.to_numpy(dtype=float)
    q_lo = m_lo.predict(X_cal)
    q_hi = m_hi.predict(X_cal)
    scores = np.maximum(q_lo - y, y - q_hi)
    q_hat = conformal_quantile(scores, alpha)
    widths = (q_hi - q_lo) + 2.0 * q_hat
    threshold = float(np.quantile(widths[np.isfinite(widths)], SELECT_Q))
    return {
        "q_hat": float(q_hat),
        "widths": widths,
        "threshold": threshold,
        "n_cal": int(len(y)),
    }


def predict_intervals(m_lo: lgb.LGBMRegressor, m_hi: lgb.LGBMRegressor,
                      X: pd.DataFrame, q_hat: float) -> dict:
    """Conformalized intervals and widths on new rows."""
    q_lo = m_lo.predict(X)
    q_hi = m_hi.predict(X)
    lo = q_lo - q_hat
    hi = q_hi + q_hat
    return {"lo": lo, "hi": hi, "widths": (hi - lo)}


def empirical_coverage(y_true: np.ndarray, lo: np.ndarray,
                       hi: np.ndarray) -> float:
    """Fraction of true values inside the intervals (diagnostic)."""
    y = np.asarray(y_true, dtype=float).ravel()
    return float(np.mean((y >= lo) & (y <= hi)))
