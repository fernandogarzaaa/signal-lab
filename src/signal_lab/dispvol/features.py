"""DISPVOL dispersion features: frozen 6-feature cross-sectional leg, point-in-time.

Feature spec (docs/DISPVOL_PREREGISTRATION.md, frozen):
- disp(t) = sample std (ddof=1) of daily log returns across the 101
  S&P 100 tickers (the SPY market leg is EXCLUDED: the bet is about
  the constituents' cross-section), over tickers with a finite return
  on day t; NaN if fewer than 50 tickers are valid.
- disp_level      = disp(t0)
- disp_5d_change  = disp(t0) - disp(t0 - 5 trading days)
- disp_z60        = (disp(t0) - mean(disp over the trailing 60 trading
                      days ending at t0)) / std(ddof=1) of the same
                      60 days
- *_x_rv1d        = each of the three interacted with the stock's own
                      daily RV leg (rv_1d from the frozen HAR set).

Point-in-time contract: every dispersion input uses closes on days
<= t0 only. disp(t) for t <= t0 is computable at t's close, so the
dispersion series built on the dev panel is point-in-time safe for
every row (verified by the truncation test: features for t0 <= cut
are identical whether computed on the full panel or on the panel
truncated at cut, mirroring the VOL point-in-time test).

Rows with fewer than 65 trading days of price history ending at t0
get NaN dispersion features and are dropped by the caller with a
logged count (never filled).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.harvix import features as harvix_feat
from signal_lab.vol.universe import MARKET_LEG

DISPERSION_BASE_COLUMNS: list[str] = ["disp_level", "disp_5d_change", "disp_z60"]
DISPERSION_COLUMNS: list[str] = DISPERSION_BASE_COLUMNS + [
    f"{c}_x_rv1d" for c in DISPERSION_BASE_COLUMNS
]

# Arm column sets. HARVIX_COLUMNS is the frozen 34-feature HAR+VIX
# set; the challenger adds exactly the 6 frozen dispersion features.
BASELINE_COLUMNS: list[str] = list(harvix_feat.HARVIX_COLUMNS)
CHALLENGER_COLUMNS: list[str] = BASELINE_COLUMNS + DISPERSION_COLUMNS

# Burn-in: 65 trading days of price history ending at t0 (60
# dispersion trailing days + 5 for the change + 1 for the day-t
# return leg, per the pre-registration).
_BURN_IN = 65
# Trailing window for the dispersion z-score.
_Z_WINDOW = 60
# Trading-day lag for the 5-day dispersion change.
_CHANGE_LAG = 5
# Minimum valid tickers for a dispersion print.
_MIN_TICKERS = 50


def dispersion_series(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-trading-day cross-sectional dispersion of returns (NaN-padded).

    Uses the union exchange calendar of the panel. The SPY market leg
    is excluded. A day with fewer than 50 valid (finite) ticker
    returns prints NaN dispersion.
    """
    px = prices[prices["ticker"] != MARKET_LEG]
    if px.empty:
        raise ValueError("[dispvol] no non-market tickers in the price panel")
    cal = pd.DatetimeIndex(sorted(px["date"].unique()))
    mat = px.pivot_table(index="date", columns="ticker",
                         values="adj_close", aggfunc="first").reindex(cal)
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(mat.to_numpy(dtype=float))
    r = np.diff(logc, axis=0, prepend=np.full((1, logc.shape[1]), np.nan))
    n_valid = np.isfinite(r).sum(axis=1)
    ok = n_valid >= _MIN_TICKERS
    disp = np.full(len(cal), np.nan)
    if ok.any():
        disp[ok] = np.nanstd(r[ok], axis=1, ddof=1)
    return pd.DataFrame({"disp_date": cal, "disp": disp})


def _per_day_disp_features(disp: np.ndarray) -> dict[str, np.ndarray]:
    """Full-length per-trading-day dispersion features (NaN-padded)."""
    n = len(disp)
    level = np.full(n, np.nan)
    change5 = np.full(n, np.nan)
    z60 = np.full(n, np.nan)
    for pos in range(_BURN_IN - 1, n):
        window = disp[pos - _Z_WINDOW + 1:pos + 1]
        if not np.all(np.isfinite(window)):
            continue
        if not np.isfinite(disp[pos - _CHANGE_LAG]):
            continue
        level[pos] = disp[pos]
        change5[pos] = disp[pos] - disp[pos - _CHANGE_LAG]
        mu = float(np.mean(window))
        sd = float(np.std(window, ddof=1))
        if sd > 0.0 and np.isfinite(sd):
            z60[pos] = (disp[pos] - mu) / sd
    return {
        "disp_level": level,
        "disp_5d_change": change5,
        "disp_z60": z60,
    }


def add_dispersion_features(frame: pd.DataFrame,
                            prices: pd.DataFrame) -> pd.DataFrame:
    """Merge the 6 frozen dispersion features onto frame rows by t0, point-in-time.

    Each row's t0 must be a union-calendar trading day of the price
    panel (the frame is built from the same panel; a mismatch fails
    loud). The frame must already carry the frozen HAR features
    (``rv_1d`` is needed for the interactions); callers add vol
    features first, then VIX features, then dispersion features.
    """
    if "rv_1d" not in frame.columns:
        raise ValueError("[dispvol] frame lacks rv_1d; add the vol "
                         "features before the dispersion features")
    series = dispersion_series(prices)
    base = _per_day_disp_features(series["disp"].to_numpy(dtype=float))
    day_ords = np.array(
        [d.toordinal() for d in series["disp_date"]], dtype=np.int64)
    t0s = pd.to_datetime(frame["t0"]).dt.normalize()
    t0_ords = np.array([d.toordinal() for d in t0s], dtype=np.int64)
    pos = np.searchsorted(day_ords, t0_ords)
    in_range = pos < len(day_ords)
    hit = np.zeros(len(frame), dtype=bool)
    hit[in_range] = day_ords[pos[in_range]] == t0_ords[in_range]
    if not hit.all():
        bad = frame.loc[~hit, "t0"].min()
        raise ValueError(
            f"[dispvol] t0 {bad} is not a price-panel trading day; refusing")
    out = frame.reset_index(drop=True).copy()
    rv1d = out["rv_1d"].to_numpy(dtype=float)
    for c in DISPERSION_BASE_COLUMNS:
        vals = base[c][pos]
        out[c] = vals
        out[f"{c}_x_rv1d"] = vals * rv1d
    return out
