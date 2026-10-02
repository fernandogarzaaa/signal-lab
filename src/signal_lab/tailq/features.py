"""TAILQ features: the 15 frozen point-in-time features.

Frozen feature list (docs/TAILQ_PREREGISTRATION.md); every feature uses
information knowable at t0's close only:

1.  rv_5d          trailing sample std (ddof=1) of daily log returns
2.  rv_22d         over (t0-5, t0], (t0-22, t0], (t0-66, t0],
3.  rv_66d         decimal daily
4.  ret_3d_pre     sum of log returns over (t0-3, t0]
5.  abs_ret_3d_pre absolute value of (4)
6.  vix_level      VIX(t0)/100/sqrt(252) (HARVIX frozen definition)
7.  vix_5d_change  VIX(t0) - VIX(t0 - 5 VIX trading days)
8.  vix_slope      VIX3M(t0)/VIX(t0) - 1
9.  beta_252       OLS slope of the stock's daily log returns on SPY's
                   over the trailing 252 trading days ending at t0
                   (minimum 60 paired observations; fewer falls back
                   to 1.0 and is counted)
10. surprise       Surprise(%)/100 from the earnings calendar row
11. abs_surprise   absolute value of (10)
12. prev_surprise  previous quarter's Surprise(%)/100; the fixed 0.0
                   fill for first events is frozen (never data-dependent)
13. dow            day-of-week of t0 (0..4)
14. spy_ret_3d_pre sum of SPY log returns over (t0-3, t0]
15. spy_rv_22d     SPY trailing 22-day sample std (ddof=1)

Rows with NaN features (insufficient trailing history) are DROPPED
with a logged count, never filled (except the frozen 0.0 fill for
first-event prev_surprise).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

FEATURE_COLUMNS: list[str] = [
    "rv_5d", "rv_22d", "rv_66d",
    "ret_3d_pre", "abs_ret_3d_pre",
    "vix_level", "vix_5d_change", "vix_slope",
    "beta_252",
    "surprise", "abs_surprise", "prev_surprise",
    "dow",
    "spy_ret_3d_pre", "spy_rv_22d",
]

_VIX_BURN_IN = 22
_BETA_WINDOW = 252
_BETA_MIN_OBS = 60
_RV66_BURN_IN = 66


def _ticker_panels(prices: pd.DataFrame) -> dict:
    """Per ticker: sorted date ordinals, closes, log returns (rets[0]=NaN)."""
    panels = {}
    for t, g in prices.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        dates = pd.to_datetime(g["date"])
        ords = dates.dt.date.apply(lambda d: d.toordinal()).to_numpy()
        closes = g["adj_close"].to_numpy(dtype=float)
        rets = np.full_like(closes, np.nan)
        rets[1:] = np.log(closes[1:] / closes[:-1])
        panels[str(t)] = (ords, closes, rets)
    return panels


def _vix_panel(vix: pd.DataFrame):
    v = vix.sort_values("date").reset_index(drop=True)
    ords = pd.to_datetime(v["date"]).dt.date.apply(
        lambda d: d.toordinal()).to_numpy()
    return ords, v["vix"].to_numpy(dtype=float), v["vix3m"].to_numpy(dtype=float)


def build_features(events: pd.DataFrame,
                   prices: pd.DataFrame,
                   vix: pd.DataFrame,
                   log=print) -> pd.DataFrame:
    """Add the 15 frozen features to the event frame; drop NaN rows.

    Returns the events frame plus FEATURE_COLUMNS. Rows that cannot be
    featurized point-in-time are dropped with a logged count.
    """
    from signal_lab.validation import periods

    periods.check_no_confirmation(events["t0"], "tailq.build_features")
    panels = _ticker_panels(prices)
    if "SPY" not in panels:
        raise ValueError("[tailq] SPY panel missing; cannot build market leg")
    spy_ords, _, spy_rets = panels["SPY"]
    spy_ret_by_ord = {o: r for o, r in zip(spy_ords, spy_rets)
                      if np.isfinite(r)}
    vix_ords, vix_c, vix3m_c = _vix_panel(vix)

    feat_rows: list[dict] = []
    beta_fallbacks = 0
    for _, e in events.iterrows():
        ticker = str(e["ticker"])
        t0 = pd.Timestamp(e["t0"]).date()
        panel = panels.get(ticker)
        if panel is None:
            continue
        ords, _, rets = panel
        pos = int(np.searchsorted(ords, t0.toordinal(), side="left"))
        if pos >= len(ords) or ords[pos] != t0.toordinal():
            raise ValueError(
                f"[tailq] t0 {t0} not on the price calendar for {ticker}")
        if pos < _RV66_BURN_IN:
            continue
        w5 = rets[pos - 4:pos + 1]
        w22 = rets[pos - 21:pos + 1]
        w66 = rets[pos - 65:pos + 1]
        w3 = rets[pos - 2:pos + 1]
        if not (np.all(np.isfinite(w5)) and np.all(np.isfinite(w22))
                and np.all(np.isfinite(w66))):
            continue
        # VIX leg: last VIX trading day <= t0, 22-day burn-in.
        vpos = int(np.searchsorted(vix_ords, t0.toordinal(), side="right")) - 1
        if vpos < _VIX_BURN_IN - 1:
            continue
        vc, vc5, v3 = vix_c[vpos], vix_c[vpos - 5], vix3m_c[vpos]
        if not (np.isfinite(vc) and np.isfinite(vc5) and np.isfinite(v3)
                and vc > 0 and vc5 > 0 and v3 > 0):
            continue
        # Beta leg: trailing 252 trading days paired with SPY.
        bstart = max(pos - _BETA_WINDOW + 1, 1)
        stock_w = rets[bstart:pos + 1]
        day_ords = ords[bstart:pos + 1]
        spy_w = np.array([spy_ret_by_ord.get(o, np.nan) for o in day_ords])
        mask = np.isfinite(stock_w) & np.isfinite(spy_w)
        if int(mask.sum()) >= _BETA_MIN_OBS and np.std(spy_w[mask]) > 0:
            beta = float(np.cov(stock_w[mask], spy_w[mask], ddof=1)[0, 1]
                         / np.var(spy_w[mask], ddof=1))
            if not np.isfinite(beta):
                beta = 1.0
                beta_fallbacks += 1
        else:
            beta = 1.0
            beta_fallbacks += 1
        # SPY legs on SPY's own calendar.
        spos = int(np.searchsorted(spy_ords, t0.toordinal(), side="left"))
        if spos >= len(spy_ords) or spy_ords[spos] != t0.toordinal():
            continue
        if spos < 22:
            continue
        spy_w3 = spy_rets[spos - 2:spos + 1]
        spy_w22 = spy_rets[spos - 21:spos + 1]
        if not (np.all(np.isfinite(spy_w3)) and np.all(np.isfinite(spy_w22))):
            continue
        surprise = (np.nan if pd.isna(e["surprise_pct"])
                    else float(e["surprise_pct"]) / 100.0)
        prev = (0.0 if pd.isna(e["prev_surprise_pct"])
                else float(e["prev_surprise_pct"]))
        if not np.isfinite(surprise):
            continue
        feat_rows.append({
            "ticker": ticker, "t0": t0,
            "rv_5d": float(np.std(w5, ddof=1)),
            "rv_22d": float(np.std(w22, ddof=1)),
            "rv_66d": float(np.std(w66, ddof=1)),
            "ret_3d_pre": float(np.sum(w3)),
            "abs_ret_3d_pre": float(abs(np.sum(w3))),
            "vix_level": float(vc / 100.0 / math.sqrt(252.0)),
            "vix_5d_change": float(vc - vc5),
            "vix_slope": float(v3 / vc - 1.0),
            "beta_252": beta,
            "surprise": surprise,
            "abs_surprise": float(abs(surprise)),
            "prev_surprise": prev,
            "dow": int(t0.weekday()),
            "spy_ret_3d_pre": float(np.sum(spy_w3)),
            "spy_rv_22d": float(np.std(spy_w22, ddof=1)),
        })
    feats = pd.DataFrame(feat_rows)
    if len(feats) == 0:
        raise ValueError("[tailq] feature build produced zero rows")
    merged = events.copy()
    merged["t0"] = pd.to_datetime(merged["t0"]).dt.date
    feats["t0"] = pd.to_datetime(feats["t0"]).dt.date
    merged = merged.merge(feats, on=["ticker", "t0"], how="inner",
                         validate="one_to_one")
    dropped = len(events) - len(merged)
    log(f"[tailq] features: {len(merged)} rows "
        f"({dropped} dropped for insufficient history); "
        f"beta fallbacks: {beta_fallbacks}")
    periods.check_no_confirmation(merged["t0"], "tailq.build_features")
    return merged.reset_index(drop=True)
