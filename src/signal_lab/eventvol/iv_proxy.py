"""EVENTVOL iv_proxy baseline: the market's forward-looking vol forecast.

iv_proxy(ticker, t0) = sqrt((beta(i,t0) x VIX_d(t0))^2 + idio_var(i,t0)),
decimal daily, where VIX_d = VIX(t0)/100/sqrt(252) and (beta, idio_var)
come from the trailing 252-trading-day market-model regression of the
stock's daily log returns on SPY's (min 60 observations).

Per-stock historical implied volatility is not obtainable from yfinance
(option chains are current-only; verified 2026-10-02), so this
market-model translation of index IV is the closest honest proxy for
the market's own forecast (see docs/EVENTVOL_PREREGISTRATION.md STEP 0).
The variance decomposition is the textbook market-model forecast: the
market prices the systematic leg (VIX), the idiosyncratic leg comes
from trailing residuals. It is the PRIMARY baseline: the campaign
question is whether anything adds to what the market already prices.

DEVIATIONS from the pre-registration's literal text (both logged in the
results doc; the first two evaluation runs are VOID):
- The pre-reg wrote beta x VIX/100/sqrt(252) with no idiosyncratic leg.
  For low-beta stocks (utilities, staples: beta ~ 0.0-0.05) that formula
  underforecasts total volatility by an order of magnitude (it prices
  only the systematic leg), punishing the baseline astronomically on
  QLIKE through no fault of the market's forecast. The variance
  decomposition implements the pre-reg's stated intent ("the market's
  forward-looking vol forecast or the closest honest proxy").
- The pre-reg specified the beta = 1.0 fallback only for <60 trailing
  observations. A non-positive trailing beta makes the market-model
  translation uninformative; the fallback (beta = 1.0, idio_var = 0,
  i.e. the raw index forecast) now also covers non-positive beta,
  non-finite beta, and degenerate regressors. The pre-reg already
  establishes beta = 1.0 as the fallback for uninformative beta
  estimates.

Fallbacks (counted, never hidden):
- beta estimation uninformative (< 60 obs, beta <= 0, non-finite, or
  degenerate SPY window): beta = 1.0, idio_var = 0.0 (raw VIX).
- missing VIX close at t0: most recent prior trading day's close.
Forecasts are floored at 1e-6 (QLIKE needs strictly positive variance
forecasts; the floor is an interface guard, never binding on real data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.eventvol import vix as vix_mod
from signal_lab.vol.universe import MARKET_LEG

FORECAST_FLOOR = 1e-6

BETA_WINDOW = 252
BETA_MIN_OBS = 60


def market_model_moments(stock_returns: np.ndarray, spy_returns: np.ndarray,
                         t0_pos: int, window: int = BETA_WINDOW,
                         min_obs: int = BETA_MIN_OBS
                         ) -> tuple[float, float, bool]:
    """Trailing market-model moments for the iv_proxy baseline.

    ``stock_returns`` and ``spy_returns`` must be aligned on the same
    (union) calendar; ``t0_pos`` indexes that calendar. Returns
    (beta, idio_var, used_fallback): the OLS beta vs SPY and the daily
    residual (idiosyncratic) variance over the trailing window ending at
    t0. Falls back to (1.0, 0.0, True) when the regression is
    uninformative (< min_obs paired observations, non-positive or
    non-finite beta, degenerate SPY window); the fallback is counted by
    the caller, never hidden.
    """
    lo = max(0, t0_pos - window + 1)
    x = spy_returns[lo:t0_pos + 1]
    y = stock_returns[lo:t0_pos + 1]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < min_obs:
        return 1.0, 0.0, True
    xm = x - x.mean()
    denom = float(np.dot(xm, xm))
    if denom <= 0:
        return 1.0, 0.0, True
    beta = float(np.dot(xm, y - y.mean()) / denom)
    if not np.isfinite(beta) or beta <= 0:
        return 1.0, 0.0, True
    resid = (y - y.mean()) - beta * xm
    idio_var = float(np.var(resid, ddof=1))
    if not np.isfinite(idio_var) or idio_var < 0:
        return 1.0, 0.0, True
    return beta, idio_var, False


def forecast_iv_proxy(rows: pd.DataFrame,
                      prices: pd.DataFrame,
                      vix: pd.DataFrame,
                      log=print) -> tuple[np.ndarray, dict[str, int]]:
    """iv_proxy forecast per row of ``rows`` (needs ticker, t0).

    Returns (forecasts, counts) where counts holds n_beta_fallback and
    n_vix_fallback. Raises on NaN output; refusing to score is the
    fail-loud contract.
    """
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    union_ords = np.array(
        sorted(pd.to_datetime(prices["date"]).dt.date
               .apply(lambda d: d.toordinal()).unique()),
        dtype=np.int64)
    pos_of = {o: i for i, o in enumerate(union_ords)}
    aligned: dict[str, np.ndarray] = {}
    for ticker, grp in prices.groupby("ticker"):
        t = str(ticker)
        g = grp.sort_values("date").reset_index(drop=True)
        dates = pd.to_datetime(g["date"]).dt.date.to_numpy()
        logc = np.log(g["adj_close"].to_numpy(dtype=float))
        r = np.empty(len(g))
        r[0] = np.nan
        r[1:] = logc[1:] - logc[:-1]
        a = np.full(len(union_ords), np.nan)
        for o, rv in zip((d.toordinal() for d in dates), r):
            a[pos_of[o]] = rv
        aligned[t] = a
    if MARKET_LEG not in aligned:
        raise ValueError(f"market leg {MARKET_LEG} missing from price panel")
    spy_r = aligned[MARKET_LEG]

    fc = np.empty(len(rows))
    n_beta_fallback = 0
    n_vix_fallback = 0
    for i, (_, row) in enumerate(rows.iterrows()):
        t = str(row["ticker"])
        t0 = pd.Timestamp(row["t0"]).date()
        upos = pos_of[t0.toordinal()]
        beta, idio_var, fb = market_model_moments(aligned[t], spy_r, upos)
        n_beta_fallback += int(fb)
        vix_close, _, vfb = vix_mod.vix_at(t0, vix)
        n_vix_fallback += int(vfb)
        vix_d = vix_close / 100.0 / np.sqrt(vix_mod.TRADING_DAYS_PER_YEAR)
        fc[i] = np.sqrt((beta * vix_d) ** 2 + idio_var)
    if int(np.isnan(fc).sum()):
        raise ValueError("[iv_proxy] NaN forecasts; refusing to score")
    fc = np.maximum(fc, FORECAST_FLOOR)
    counts = {"n_beta_fallback": n_beta_fallback,
              "n_vix_fallback": n_vix_fallback,
              "n_rows": len(rows)}
    log(f"[iv_proxy] {len(rows)} forecasts, beta fallbacks "
        f"{n_beta_fallback}, VIX fallbacks {n_vix_fallback}")
    return fc, counts
