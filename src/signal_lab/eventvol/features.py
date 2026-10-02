"""EVENTVOL features: HAR+VIX and challenger feature sets, point-in-time.

- HAR legs: the frozen 31-column VOL feature set
  (signal_lab.vol.features.FEATURE_COLUMNS), computed as of t0's close.
- VIX features: vix_level, vix_5d_change, vix_runup_22d, vix_slope
  (signal_lab.eventvol.vix), as of t0's close.
- Event features (challenger only; Ederington-Lee 1996 IV
  creation/resolution cycle proxies, all knowable at t0's close):
  event_day_absret = |r(t0)|, event_day_ret = r(t0),
  trailing_surprise = previous quarter's Surprise(%)/100
  (the current quarter's surprise is excluded: not knowable at t0's
  close for after-close announcements),
  beta_252 = trailing 252-trading-day OLS beta vs SPY.

Point-in-time contract: every trailing window ends at t0; nothing past
t0's close enters a feature. Insufficient history yields NaN, NEVER
filled. Callers drop NaN-feature rows explicitly via
``drop_nan_features`` (the drop count is reported).

Feature sets (frozen in docs/EVENTVOL_PREREGISTRATION.md):
    HARVIX_FEATURES    = HAR(31) + VIX(4)
    CHALLENGER_FEATURES = HARVIX_FEATURES + EVENT(4)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.eventvol import vix as vix_mod
from signal_lab.vol import features as vol_features
from signal_lab.vol.universe import MARKET_LEG

HARVIX_FEATURES: list[str] = (
    list(vol_features.FEATURE_COLUMNS) + list(vix_mod.VIX_FEATURE_COLUMNS)
)

EVENT_FEATURE_COLUMNS: list[str] = [
    "event_day_absret",
    "event_day_ret",
    "trailing_surprise",
    "beta_252",
]

CHALLENGER_FEATURES: list[str] = HARVIX_FEATURES + EVENT_FEATURE_COLUMNS

BETA_WINDOW = 252
BETA_MIN_OBS = 60


def trailing_beta(stock_returns: np.ndarray, spy_returns: np.ndarray,
                  t0_pos: int, window: int = BETA_WINDOW,
                  min_obs: int = BETA_MIN_OBS) -> tuple[float, bool]:
    """OLS beta of stock on SPY over the trailing window ending at t0.

    ``stock_returns`` and ``spy_returns`` must be aligned on the same
    (union) calendar; ``t0_pos`` indexes that calendar. Returns (beta,
    used_fallback). Falls back to 1.0 when fewer than ``min_obs``
    finite paired observations exist; the fallback is counted by the
    caller, never hidden.
    """
    lo = max(0, t0_pos - window + 1)
    x = spy_returns[lo:t0_pos + 1]
    y = stock_returns[lo:t0_pos + 1]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < min_obs:
        return 1.0, True
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 1.0, True
    beta = float(np.dot(x, y - y.mean()) / denom)
    if not np.isfinite(beta):
        return 1.0, True
    return beta, False


def add_features(frame: pd.DataFrame, prices: pd.DataFrame,
                 vix: pd.DataFrame, log=print) -> pd.DataFrame:
    """Merge HAR + VIX + event features onto event-frame rows.

    ``frame``: output of earnings.build_event_frame (must carry
    prev_surprise_pct). Returns the frame with HARVIX_FEATURES +
    EVENT_FEATURE_COLUMNS columns appended.
    """
    har = vol_features.add_features(frame, prices)
    if MARKET_LEG not in set(prices["ticker"]):
        raise ValueError(f"market leg {MARKET_LEG} missing from price panel")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    ret_by_ticker: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ticker, grp in prices.groupby("ticker"):
        t = str(ticker)
        g = grp.sort_values("date").reset_index(drop=True)
        dates = pd.to_datetime(g["date"]).dt.date.to_numpy()
        logc = np.log(g["adj_close"].to_numpy(dtype=float))
        r = np.empty(len(g))
        r[0] = np.nan
        r[1:] = logc[1:] - logc[:-1]
        ret_by_ticker[t] = (np.array([d.toordinal() for d in dates]), r)
    # Align both legs on the union calendar so the trailing beta window
    # is date-matched, not position-matched.
    union_ords = np.array(
        sorted({o for ords, _ in ret_by_ticker.values() for o in ords}),
        dtype=np.int64)
    pos_of = {o: i for i, o in enumerate(union_ords)}
    aligned: dict[str, np.ndarray] = {}
    for t, (ords, r) in ret_by_ticker.items():
        a = np.full(len(union_ords), np.nan)
        for o, rv in zip(ords, r):
            a[pos_of[o]] = rv
        aligned[t] = a
    spy_r = aligned[MARKET_LEG]

    feat_rows: list[dict] = []
    n_beta_fallback = 0
    for _, row in frame.iterrows():
        t = str(row["ticker"])
        t0 = pd.Timestamp(row["t0"]).date()
        feats = dict(vix_mod.vix_features_at(t0, vix))
        upos = pos_of[t0.toordinal()]
        r = aligned[t]
        r_t0 = float(r[upos])
        feats["event_day_ret"] = r_t0
        feats["event_day_absret"] = abs(r_t0) if np.isfinite(r_t0) else np.nan
        ps = row["prev_surprise_pct"]
        feats["trailing_surprise"] = (np.nan if pd.isna(ps) else float(ps))
        beta, fb = trailing_beta(r, spy_r, upos)
        feats["beta_252"] = beta
        n_beta_fallback += int(fb)
        feat_rows.append(feats)
    ev = pd.DataFrame(feat_rows)
    log(f"[features] beta fallbacks (beta=1.0): {n_beta_fallback}/{len(frame)}")
    for c in HARVIX_FEATURES + EVENT_FEATURE_COLUMNS:
        if c not in har.columns:
            har[c] = ev[c].to_numpy() if c in ev.columns else np.nan
    # HAR columns came from vol_features.add_features; VIX/event columns
    # from ev. Overwrite with ev's authoritative values for those columns.
    for c in list(vix_mod.VIX_FEATURE_COLUMNS) + EVENT_FEATURE_COLUMNS:
        har[c] = ev[c].to_numpy()
    return har


def drop_nan_features(frame: pd.DataFrame,
                      columns: list[str] | None = None,
                      log=print) -> tuple[pd.DataFrame, int]:
    """Drop rows with NaN in any feature column; return (clean, n_dropped)."""
    cols = columns or CHALLENGER_FEATURES
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"feature columns missing: {missing}")
    mask = frame[cols].notna().all(axis=1).to_numpy()
    n_dropped = int(len(frame) - mask.sum())
    if n_dropped:
        log(f"[features] dropped {n_dropped}/{len(frame)} rows with NaN "
            f"features")
    return frame.loc[mask].reset_index(drop=True), n_dropped
