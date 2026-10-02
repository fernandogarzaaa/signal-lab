"""EVENTVOL iv_proxy baseline: the market's forward-looking vol forecast.

iv_proxy(ticker, t0) = beta(i, t0) x VIX(t0) / 100 / sqrt(252),
decimal daily. beta(i, t0) is the trailing 252-trading-day OLS beta vs
SPY (signal_lab.eventvol.features.trailing_beta); VIX(t0) is the CBOE
VIX close at t0 (signal_lab.eventvol.vix.vix_at).

Per-stock historical implied volatility is not obtainable from yfinance
(option chains are current-only; verified 2026-10-02), so this index-IV
translation is the closest honest proxy for the market's own forecast
(see docs/EVENTVOL_PREREGISTRATION.md STEP 0). It is the PRIMARY
baseline: the campaign question is whether anything adds to what the
market already prices.

Fallbacks (counted, never hidden):
- beta estimation with < 60 observations: beta = 1.0 (raw VIX).
- missing VIX close at t0: most recent prior trading day's close.
Forecasts are floored at 1e-6 (QLIKE needs strictly positive variance
forecasts; the floor is an interface guard, never binding on real data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.eventvol import features as feat_mod
from signal_lab.eventvol import vix as vix_mod
from signal_lab.vol.universe import MARKET_LEG

FORECAST_FLOOR = 1e-6


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
        beta, fb = feat_mod.trailing_beta(aligned[t], spy_r, upos)
        n_beta_fallback += int(fb)
        vix_close, _, vfb = vix_mod.vix_at(t0, vix)
        n_vix_fallback += int(vfb)
        fc[i] = beta * vix_close / 100.0 / np.sqrt(
            vix_mod.TRADING_DAYS_PER_YEAR)
    if int(np.isnan(fc).sum()):
        raise ValueError("[iv_proxy] NaN forecasts; refusing to score")
    fc = np.maximum(fc, FORECAST_FLOOR)
    counts = {"n_beta_fallback": n_beta_fallback,
              "n_vix_fallback": n_vix_fallback,
              "n_rows": len(rows)}
    log(f"[iv_proxy] {len(rows)} forecasts, beta fallbacks "
        f"{n_beta_fallback}, VIX fallbacks {n_vix_fallback}")
    return fc, counts
