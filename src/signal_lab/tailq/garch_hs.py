"""GARCH-filtered historical-simulation VaR (secondary baseline).

Per (ticker, t0):
- GARCH(1,1) by MLE on the trailing 252 trading days through t0
  (reuse src/signal_lab/vol/garch.py, unchanged).
- In-sample conditional variance path from the fitted recursion
  (identical math to vol/garch.py's recursion), standardized
  residuals z_t = r_t / sigma_t.
- q05 = empirical 5% quantile of z (numpy quantile).
- 1-day VaR = sigma_{t0+1|t0} * q05; 3-day VaR = 1-day VaR * sqrt(3)
  (the same frozen horizon scaling as the other history-based arms).
- Fit failures fall back to the naive empirical-quantile forecast and
  are COUNTED, never hidden.
"""

from __future__ import annotations

import math

import numpy as np

from signal_lab.vol.garch import ESTIMATION_WINDOW, fit_garch11

TAU = 0.05
HORIZON_SCALE = math.sqrt(3.0)


def _sigma_path(omega: float, alpha: float, beta: float,
                r: np.ndarray, var0: float) -> np.ndarray:
    """In-sample conditional variances; same recursion as vol/garch.py."""
    n = r.shape[0]
    s2 = np.empty(n)
    s2[0] = var0
    r2 = r * r
    for t in range(1, n):
        s2[t] = omega + alpha * r2[t - 1] + beta * s2[t - 1]
    return s2


def _fallback(returns: np.ndarray, window: int, tau: float) -> tuple[float, bool]:
    from signal_lab.tailq.models import naive_3d_quantile
    return naive_3d_quantile(returns, window=window, tau=tau), True


def forecast_garch_hs_3d(returns: np.ndarray, t0_pos: int,
                         window: int = ESTIMATION_WINDOW,
                         tau: float = TAU) -> tuple[float, bool]:
    """GARCH-filtered HS 3-day VaR forecast for one t0; (forecast, fell_back).

    ``returns``: full daily log-return array for the ticker; ``t0_pos``:
    index of t0 in that array (estimation uses returns through t0).
    """
    r = np.asarray(returns, dtype=float).ravel()[:t0_pos + 1]
    if r.size < window:
        return _fallback(r, window, tau)
    rw = r[-window:]
    if not np.all(np.isfinite(rw)):
        return _fallback(rw, window, tau)
    fit = fit_garch11(rw, window=window)
    if not fit.converged:
        return _fallback(rw, window, tau)
    var0 = float(np.var(rw, ddof=1))
    s2 = _sigma_path(fit.omega, fit.alpha, fit.beta, rw, var0)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return _fallback(rw, window, tau)
    z = rw / np.sqrt(s2)
    q05 = float(np.quantile(z, tau))
    sigma_1step = math.sqrt(
        fit.omega + fit.alpha * rw[-1] ** 2 + fit.beta * s2[-1])
    if not (np.isfinite(q05) and np.isfinite(sigma_1step) and sigma_1step > 0):
        return _fallback(rw, window, tau)
    return float(sigma_1step * q05 * HORIZON_SCALE), False
