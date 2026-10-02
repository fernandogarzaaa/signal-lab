"""HAR-style point-in-time price features as of t0 close only.

Own leg (per ticker) and the same legs for SPY (``spy_`` prefix):
- rv_1d   = r_{t0}^2
- rv_5d   = mean(r^2) over trailing 5 trading days ending at t0
- rv_22d  = mean(r^2) over trailing 22 trading days ending at t0
  (Corsi HAR daily / weekly / monthly legs)
- sqret_lag1..5 = r_{t0-k}^2 for k = 1..5
- parkinson_1d  = (ln(H_{t0}/L_{t0}))^2 / (4 ln 2)  (Parkinson 1980)
- parkinson_5d  = trailing-5-day mean of the daily Parkinson estimator
- logvol     = ln(V_{t0})
- dlogvol_1d = ln(V_{t0}) - ln(V_{t0-1})
- dlogvol_5d = ln(V_{t0}) - ln(mean(V) trailing 5)
- dow_0..dow_4 = one-hot day-of-week of t0 (Monday..Friday; shared, not per leg)

Point-in-time contract: every trailing window ends at t0; nothing past
t0's close enters a feature. Verified by
tests/test_vol_features.py::test_point_in_time_truncation (features for
t0 <= cut are identical whether computed on the full panel or on the
panel truncated at cut).

Insufficient history yields NaN, NEVER filled. Callers drop NaN-feature
rows explicitly via ``drop_nan_features`` (the drop count is reported).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.vol.universe import MARKET_LEG

_OWN_LEG = [
    "rv_1d", "rv_5d", "rv_22d",
    "sqret_lag1", "sqret_lag2", "sqret_lag3", "sqret_lag4", "sqret_lag5",
    "parkinson_1d", "parkinson_5d",
    "logvol", "dlogvol_1d", "dlogvol_5d",
]
_SPY_LEG = [f"spy_{c}" for c in _OWN_LEG]
_DOW = [f"dow_{i}" for i in range(5)]

FEATURE_COLUMNS: list[str] = _OWN_LEG + _SPY_LEG + _DOW

_PARKINSON_DENOM = 4.0 * np.log(2.0)


def _leg_arrays(dates: pd.DatetimeIndex, high: np.ndarray, low: np.ndarray,
                adj_close: np.ndarray, volume: np.ndarray) -> dict[str, np.ndarray]:
    """Full-length per-day feature arrays for one price leg (NaN-padded)."""
    n = len(dates)
    logc = np.log(np.where(adj_close > 0, adj_close, np.nan))
    r = np.empty(n)
    r[0] = np.nan
    r[1:] = logc[1:] - logc[:-1]
    r2 = r ** 2

    def roll_mean(x: np.ndarray, w: int) -> np.ndarray:
        return pd.Series(x).rolling(w, min_periods=w).mean().to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        park = (np.log(high / low)) ** 2 / _PARKINSON_DENOM
        logvol = np.log(np.where(volume > 0, volume, np.nan))
    mean_vol_5 = pd.Series(np.where(volume > 0, volume, np.nan)).rolling(5, min_periods=5).mean().to_numpy()

    out = {
        "rv_1d": r2,
        "rv_5d": roll_mean(r2, 5),
        "rv_22d": roll_mean(r2, 22),
        "parkinson_1d": park,
        "parkinson_5d": roll_mean(park, 5),
        "logvol": logvol,
        "dlogvol_5d": logvol - np.log(mean_vol_5),
    }
    for k in range(1, 6):
        out[f"sqret_lag{k}"] = pd.Series(r2).shift(k).to_numpy()
    out["dlogvol_1d"] = pd.Series(logvol).diff(1).to_numpy()
    return out


def add_features(frame: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Merge point-in-time features onto frame rows by (ticker, t0)."""
    if MARKET_LEG not in set(prices["ticker"]):
        raise ValueError(f"market leg {MARKET_LEG} missing from price panel")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    legs: dict[str, dict[str, np.ndarray]] = {}
    date_index: dict[str, pd.DatetimeIndex] = {}
    for ticker, grp in prices.groupby("ticker"):
        t = str(ticker)
        dates = pd.DatetimeIndex(grp["date"])
        date_index[t] = dates
        legs[t] = _leg_arrays(
            dates,
            grp["high"].to_numpy(dtype=float),
            grp["low"].to_numpy(dtype=float),
            grp["adj_close"].to_numpy(dtype=float),
            grp["volume"].to_numpy(dtype=float),
        )
    spy_arrays = legs[MARKET_LEG]
    spy_dates = date_index[MARKET_LEG]
    spy_pos = {d.date(): i for i, d in enumerate(spy_dates)}

    feat_rows: list[dict] = []
    for _, row in frame.iterrows():
        t = str(row["ticker"])
        t0 = pd.Timestamp(row["t0"]).date()
        pos = date_index[t].get_loc(pd.Timestamp(t0))
        if isinstance(pos, slice):  # pragma: no cover - dates are unique per ticker
            raise ValueError(f"non-unique date {t0} for {t}")
        pos = int(pos)
        feats: dict[str, float] = {}
        for c in _OWN_LEG:
            feats[c] = float(legs[t][c][pos])
        spos = spy_pos.get(t0)
        for c, sc in zip(_OWN_LEG, _SPY_LEG):
            feats[sc] = float(spy_arrays[c][spos]) if spos is not None else np.nan
        dow = pd.Timestamp(t0).weekday()
        for i in range(5):
            feats[f"dow_{i}"] = 1.0 if dow == i else 0.0
        feat_rows.append(feats)
    feats_df = pd.DataFrame(feat_rows, columns=FEATURE_COLUMNS, index=frame.index)
    return pd.concat([frame.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)


def drop_nan_features(df: pd.DataFrame,
                      feature_columns: list[str] = FEATURE_COLUMNS,
                      log=print) -> tuple[pd.DataFrame, int]:
    """Explicit caller-side drop of rows with NaN features.

    NaN means insufficient trailing history (burn-in); it is never
    filled. Returns (clean_frame, n_dropped) and logs the count.
    """
    mask = df[feature_columns].notna().all(axis=1)
    n_dropped = int((~mask).sum())
    log(f"[features] dropped {n_dropped}/{len(df)} rows with NaN features (insufficient history)")
    return df[mask].reset_index(drop=True), n_dropped
