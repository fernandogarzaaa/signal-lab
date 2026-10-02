"""Engle-Manganelli (2004) CAViaR: symmetric absolute value specification.

The canonical SAV CAViaR (the paper's leading specification):

    f_t(beta) = b1 + b2 * f_{t-1} + b3 * |r_{t-1}|

Estimated per (ticker, t0) by minimizing the quantile check function
at tau = 0.05 over the trailing 252 trading-day daily log returns
ending at t0:

    sum_t rho_tau(r_t - f_t(beta)),  rho_tau(u) = u * (tau - 1[u < 0]).

Frozen estimation protocol (docs/TAILQ_PREREGISTRATION.md):
- f_1 initialized at the empirical 5% quantile of the window.
- scipy differential_evolution (seed=7, maxiter=100), bounds
  b1 in [3*q05, 0], b2 in [0.0, 0.999] (stationarity guard),
  b3 in [0.0, 2.0]; Nelder-Mead polish from the DE solution.
- Minimum 100 observations; fewer is a fit failure.
- 1-day VaR forecast f_{T+1} = b1 + b2*f_T + b3*|r_T|.
- Frozen horizon adaptation: the canonical model is a 1-day model;
  the 3-day VaR forecast is f_{T+1} * sqrt(3) (i.i.d. variance-time
  scaling), applied identically to all history-based arms.
- Fit failures fall back to the naive empirical-quantile forecast and
  are COUNTED, never hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

TAU = 0.05
ESTIMATION_WINDOW = 252
MIN_OBS = 100
HORIZON_SCALE = math.sqrt(3.0)


@dataclass(frozen=True)
class CaviarFit:
    b1: float
    b2: float
    b3: float
    f_last: float  # last in-sample conditional quantile f_T
    n: int
    converged: bool
    detail: str  # "ok" or the failure reason; failures are counted by callers


def _check_loss(resid: np.ndarray, tau: float) -> float:
    return float(np.sum(resid * (tau - (resid < 0).astype(float))))


def _forward_path(beta: np.ndarray, r: np.ndarray, f0: float) -> np.ndarray:
    b1, b2, b3 = beta
    n = r.shape[0]
    f = np.empty(n)
    f[0] = f0
    abs_r = np.abs(r)
    for t in range(1, n):
        f[t] = b1 + b2 * f[t - 1] + b3 * abs_r[t - 1]
    return f


def _objective(beta: np.ndarray, r: np.ndarray, f0: float, tau: float) -> float:
    f = _forward_path(beta, r, f0)
    if not np.all(np.isfinite(f)):
        return np.inf
    return _check_loss(r - f, tau)


def fit_caviar_sav(returns: np.ndarray,
                   tau: float = TAU,
                   window: int = ESTIMATION_WINDOW,
                   seed: int = 7) -> CaviarFit:
    """SAV CAViaR fit on the last ``window`` returns of ``returns``."""
    r = np.asarray(returns, dtype=float).ravel()
    if r.size < window:
        return CaviarFit(np.nan, np.nan, np.nan, np.nan, r.size, False,
                         f"short-window:{r.size}<{window}")
    rw = r[-window:]
    if not np.all(np.isfinite(rw)):
        return CaviarFit(np.nan, np.nan, np.nan, np.nan, window, False,
                         "non-finite-returns")
    if rw.size < MIN_OBS:
        return CaviarFit(np.nan, np.nan, np.nan, np.nan, rw.size, False,
                         f"short-window:{rw.size}<{MIN_OBS}")
    q05 = float(np.quantile(rw, tau))
    if not np.isfinite(q05) or q05 >= 0:
        return CaviarFit(np.nan, np.nan, np.nan, np.nan, window, False,
                         f"degenerate-quantile:{q05}")
    bounds = [(3.0 * q05, 0.0), (0.0, 0.999), (0.0, 2.0)]
    try:
        de = differential_evolution(
            _objective, bounds, args=(rw, q05, tau),
            seed=seed, maxiter=100, tol=1e-7, polish=False,
        )
    except Exception as exc:
        return CaviarFit(np.nan, np.nan, np.nan, np.nan, window, False,
                         f"de-error:{type(exc).__name__}")
    if not np.isfinite(de.fun):
        return CaviarFit(np.nan, np.nan, np.nan, np.nan, window, False,
                         "de-non-finite")
    try:
        nm = minimize(_objective, de.x, args=(rw, q05, tau),
                      method="Nelder-Mead",
                      options={"maxiter": 500, "xatol": 1e-8, "fatol": 1e-8})
        x = nm.x if (nm.success and np.isfinite(nm.fun)
                     and nm.fun <= de.fun) else de.x
    except Exception:
        x = de.x
    b1, b2, b3 = (float(v) for v in x)
    f = _forward_path(np.array([b1, b2, b3]), rw, q05)
    if not np.all(np.isfinite(f)):
        return CaviarFit(np.nan, np.nan, np.nan, np.nan, window, False,
                         "non-finite-path")
    return CaviarFit(b1=b1, b2=b2, b3=b3, f_last=float(f[-1]), n=window,
                     converged=True, detail="ok")


def forecast_caviar_3d(fit: CaviarFit, r_last: float) -> float:
    """3-day VaR forecast from a converged fit (raises otherwise)."""
    if not fit.converged:
        raise ValueError(f"cannot forecast from a failed fit ({fit.detail})")
    f_next = fit.b1 + fit.b2 * fit.f_last + fit.b3 * abs(float(r_last))
    if not np.isfinite(f_next):
        raise ValueError(f"non-finite CAViaR 1-day forecast {f_next}")
    return float(f_next * HORIZON_SCALE)


def forecast_at_t0(returns: np.ndarray, t0_pos: int,
                   window: int = ESTIMATION_WINDOW,
                   tau: float = TAU,
                   seed: int = 7) -> tuple[float, bool]:
    """CAViaR 3-day VaR forecast for one t0; (forecast, fell_back).

    ``returns``: full daily log-return array for the ticker; ``t0_pos``:
    index of t0 in that array (estimation uses returns through t0).
    """
    rw = np.asarray(returns, dtype=float).ravel()[:t0_pos + 1]
    fit = fit_caviar_sav(rw, tau=tau, window=window, seed=seed)
    if fit.converged:
        try:
            return forecast_caviar_3d(fit, rw[-1]), False
        except ValueError:
            pass
    from signal_lab.tailq.models import naive_3d_quantile
    return naive_3d_quantile(rw, window=window, tau=tau), True
