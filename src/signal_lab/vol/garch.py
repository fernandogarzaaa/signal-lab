"""GARCH(1,1) baseline: MLE on trailing 252 trading days, analytic term structure.

Frozen spec (docs/VOL_PREREGISTRATION.md):
- Fit by maximum likelihood (Gaussian innovations, zero-mean: the daily
  mean is negligible next to daily variance and the spec's recursion is
  written in r_t^2) on the trailing 252 trading days of daily log returns
  through t0.
- Forecast = analytic 5-day term structure
      E[sigma^2_{t+h}|F_t] = sbar^2 + (alpha+beta)^(h-1) * (s2_1 - sbar^2),
  summed over h = 1..5 and square-rooted to realized vol, where
  sbar^2 = omega / (1 - alpha - beta) and
  s2_1 = omega + alpha * r_t^2 + beta * s2_t (one-step-ahead from the last
  fitted conditional variance).
- Fit failures (non-convergence, alpha + beta >= 1, non-finite or
  non-positive variances, short estimation window) fall back to the naive
  persistence forecast and are COUNTED, never hidden.

Implemented with scipy.optimize L-BFGS-B and an analytic gradient (no new
dependencies; ``arch`` is not installed and CI installs only from
packages/cli/engine/requirements.txt).
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ESTIMATION_WINDOW = 252
HORIZON_DAYS = 5

# Two pre-declared starting points (variance-targeting style). A second
# start guards against local-optimum non-convergence; both are fixed, not
# tuned.
_STARTS = ((0.05, 0.07, 0.88), (0.10, 0.12, 0.80))  # (omega/var, alpha, beta)


@dataclass(frozen=True)
class GarchFit:
    omega: float
    alpha: float
    beta: float
    sigma2_last: float  # conditional variance of the last in-sample return
    n: int
    converged: bool
    detail: str  # "ok" or the failure reason; failures are counted by callers


def _recursion(theta: np.ndarray, r2: np.ndarray, var0: float):
    """Conditional variances and their derivatives wrt (omega, alpha, beta).

    sigma2[0] = var0 (fixed); for t >= 1:
    sigma2[t] = omega + alpha * r2[t-1] + beta * sigma2[t-1].
    """
    omega, alpha, beta = theta
    n = r2.shape[0]
    s2 = np.empty(n)
    d_om = np.empty(n)
    d_al = np.empty(n)
    d_be = np.empty(n)
    s2[0] = var0
    d_om[0] = d_al[0] = d_be[0] = 0.0
    for t in range(1, n):
        s2[t] = omega + alpha * r2[t - 1] + beta * s2[t - 1]
        d_om[t] = 1.0 + beta * d_om[t - 1]
        d_al[t] = r2[t - 1] + beta * d_al[t - 1]
        d_be[t] = s2[t - 1] + beta * d_be[t - 1]
    return s2, d_om, d_al, d_be


def _negloglik_and_grad(theta: np.ndarray, r: np.ndarray, r2: np.ndarray,
                        var0: float):
    s2, d_om, d_al, d_be = _recursion(theta, r2, var0)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return np.inf, np.zeros(3)
    inv = 1.0 / s2
    # d loglik / d sigma2[t] = -0.5 * (1/s2 - r^2/s2^2); negate for the NLL.
    w = 0.5 * (inv - r * r * inv * inv)
    nll = 0.5 * float(np.sum(np.log(2.0 * np.pi) + np.log(s2) + r * r * inv))
    grad = np.array([
        float(np.sum(w * d_om)),
        float(np.sum(w * d_al)),
        float(np.sum(w * d_be)),
    ])
    if not np.all(np.isfinite(grad)):
        return np.inf, np.zeros(3)
    return nll, grad


def _fail(detail: str, n: int) -> GarchFit:
    return GarchFit(omega=np.nan, alpha=np.nan, beta=np.nan,
                    sigma2_last=np.nan, n=n, converged=False, detail=detail)


def fit_garch11(r: np.ndarray, window: int = ESTIMATION_WINDOW) -> GarchFit:
    """MLE fit of GARCH(1,1) on the last ``window`` returns of ``r``."""
    r = np.asarray(r, dtype=float).ravel()
    if r.size < window:
        return _fail(f"short-window:{r.size}<{window}", r.size)
    rw = r[-window:]
    r2 = rw * rw
    if not np.all(np.isfinite(rw)):
        return _fail("non-finite-returns", window)
    var0 = float(np.var(rw, ddof=1))
    if not np.isfinite(var0) or var0 <= 0:
        return _fail("degenerate-variance", window)

    bounds = [(1e-12, 10.0 * var0), (1e-8, 0.999), (1e-8, 0.999)]
    best = None
    for om_frac, a0, b0 in _STARTS:
        x0 = np.array([om_frac * var0, a0, b0])
        try:
            res = minimize(
                _negloglik_and_grad, x0, jac=True, args=(rw, r2, var0),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 500},
            )
        except Exception:  # optimizer-level blowup -> try next start
            continue
        if not res.success or not np.isfinite(res.fun):
            continue
        omega, alpha, beta = (float(v) for v in res.x)
        if alpha + beta >= 1.0:
            continue
        if best is None or res.fun < best[0]:
            best = (float(res.fun), omega, alpha, beta)
    if best is None:
        return _fail("non-convergence", window)
    _, omega, alpha, beta = best
    s2, _, _, _ = _recursion(np.array([omega, alpha, beta]), r2, var0)
    sigma2_last = float(s2[-1])
    if not np.isfinite(sigma2_last) or sigma2_last <= 0:
        return _fail("non-positive-variance", window)
    return GarchFit(omega=omega, alpha=alpha, beta=beta,
                    sigma2_last=sigma2_last, n=window,
                    converged=True, detail="ok")


def forecast_garch11(fit: GarchFit, r_last: float,
                     horizon: int = HORIZON_DAYS) -> float:
    """Analytic h-step term-structure forecast, square-rooted to vol."""
    if not fit.converged:
        raise ValueError(f"cannot forecast from a failed fit ({fit.detail})")
    if horizon != HORIZON_DAYS:
        raise ValueError(f"frozen horizon is {HORIZON_DAYS}, got {horizon}")
    persistence = fit.alpha + fit.beta
    sbar2 = fit.omega / (1.0 - persistence)
    s2_1 = fit.omega + fit.alpha * r_last ** 2 + fit.beta * fit.sigma2_last
    total = 0.0
    for h in range(1, horizon + 1):
        total += sbar2 + persistence ** (h - 1) * (s2_1 - sbar2)
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"non-positive forecast variance {total}")
    return math.sqrt(total)


def naive_forecast(returns: np.ndarray, t0_pos: int,
                   horizon: int = HORIZON_DAYS) -> float:
    """Persistence baseline: trailing realized_vol_5d as of t0.

    Uses the 5 fully-observed daily returns ending at t0: r_{t0-4..t0}.
    """
    window = returns[t0_pos - horizon + 1:t0_pos + 1]
    if window.shape[0] < horizon or not np.all(np.isfinite(window)):
        return np.nan
    return float(np.std(window, ddof=1))


def forecast_at_t0(returns: np.ndarray, t0_pos: int,
                   window: int = ESTIMATION_WINDOW,
                   horizon: int = HORIZON_DAYS) -> tuple[float, bool]:
    """GARCH(1,1) forecast for one t0; (forecast, fell_back_to_naive).

    ``returns``: full daily log-return array for the ticker; ``t0_pos``:
    index of t0 in that array (estimation uses returns through t0).
    """
    r_last = float(returns[t0_pos])
    naive = naive_forecast(returns, t0_pos, horizon)
    fit = fit_garch11(returns[:t0_pos + 1], window=window)
    if not fit.converged:
        return naive, True
    try:
        fc = forecast_garch11(fit, r_last, horizon)
    except ValueError:
        return naive, True
    if not np.isfinite(fc) or fc <= 0:
        return naive, True
    return fc, False


def _ticker_task(args) -> list[tuple[float, bool]]:
    """One task per ticker: forecasts for all its t0s. Pickle-friendly."""
    dates_ord, returns, t0_ords, window, horizon = args
    out: list[tuple[float, bool]] = []
    for t0o in t0_ords:
        pos = int(np.searchsorted(dates_ord, t0o, side="left"))
        if pos >= len(dates_ord) or dates_ord[pos] != t0o:
            raise ValueError(f"t0 ordinal {t0o} not on the ticker's date grid")
        out.append(forecast_at_t0(returns, pos, window=window, horizon=horizon))
    return out


def forecast_panel(rows: pd.DataFrame,
                   returns_by_ticker: dict[str, pd.Series],
                   window: int = ESTIMATION_WINDOW,
                   horizon: int = HORIZON_DAYS,
                   n_jobs: int = 1,
                   log=print) -> pd.DataFrame:
    """GARCH(1,1) forecasts for every (ticker, t0) in ``rows``.

    Returns a DataFrame aligned to ``rows`` with columns
    ``garch_vol`` and ``garch_fallback``. Deterministic regardless of
    ``n_jobs`` (per-ticker tasks reassembled in input order).
    """
    rows = rows.reset_index(drop=True)
    t0_dates = pd.to_datetime(rows["t0"]).dt.date
    by_ticker: dict[str, list[int]] = {}
    for i, (t, d) in enumerate(zip(rows["ticker"].astype(str), t0_dates)):
        by_ticker.setdefault(t, []).append(i)

    tasks = []
    order: list[tuple[str, list[int]]] = []
    for t, idxs in by_ticker.items():
        if t not in returns_by_ticker:
            raise ValueError(f"no return series for ticker {t}")
        rs = returns_by_ticker[t].sort_index()
        dates_ord = np.array([d.toordinal() for d in rs.index.date], dtype=np.int64)
        rets = rs.to_numpy(dtype=float)
        t0_ords = np.array([d.toordinal() for d in t0_dates.iloc[idxs]], dtype=np.int64)
        tasks.append((dates_ord, rets, t0_ords, window, horizon))
        order.append((t, idxs))

    if n_jobs == 1:
        results = [_ticker_task(a) for a in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_ticker_task, tasks))

    vols = np.empty(len(rows))
    fallbacks = np.empty(len(rows), dtype=bool)
    for (_, idxs), res in zip(order, results):
        for i, (fc, fb) in zip(idxs, res):
            vols[i] = fc
            fallbacks[i] = fb
    n_fb = int(fallbacks.sum())
    n_nan = int(np.isnan(vols).sum())
    if n_nan:
        raise ValueError(
            f"[garch] {n_nan} NaN forecasts (no naive value available); "
            "refusing to score NaN forecasts"
        )
    log(f"[garch] {len(rows)} forecasts, {n_fb} naive fallbacks "
        f"({100.0 * n_fb / max(len(rows), 1):.2f}%)")
    out = rows[["ticker", "t0"]].copy()
    out["garch_vol"] = vols
    out["garch_fallback"] = fallbacks
    return out
