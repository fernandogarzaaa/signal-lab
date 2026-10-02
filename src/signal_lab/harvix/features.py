"""HARVIX VIX features: frozen 3-feature market leg, point-in-time.

Feature spec (docs/HARVIX_PREREGISTRATION.md, frozen):
- vix_level     = VIX(t0) / 100 / sqrt(252)            (decimal daily)
- vix_5d_change = VIX(t0) - VIX(t0 - 5 VIX trading days) (VIX points)
- vix_slope     = VIX3M(t0) / VIX(t0) - 1               (dimensionless)

Point-in-time contract: every feature uses VIX trading-day closes on
days <= t0 only. Rows with fewer than 22 VIX trading days of history
ending at t0, or with non-positive VIX/VIX3M closes at t0, get NaN and
are dropped by the caller with a logged count (never filled).

The per-day VIX math is reused from ``signal_lab.eventvol.vix``
(``download_vix``, and the same close-lookup semantics); only the
frozen 3-column selection and the frame merge are new here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.eventvol import vix as vix_mod
from signal_lab.vol import features as vol_features

VIX_FEATURE_COLUMNS: list[str] = ["vix_level", "vix_5d_change", "vix_slope"]

# Burn-in: need 22 VIX trading days ending at t0 (the 5-day change plus a
# stable mapping window, matching the price legs' 22-day burn-in).
_VIX_BURN_IN = 22


def load_vix(cache_path, force_download: bool = False, log=print) -> pd.DataFrame:
    """Load (or download, fail-loud) the ^VIX/^VIX3M daily closes.

    The download FAILS LOUDLY on failure or empty results: the campaign
    cannot run without the term-structure slope.
    """
    return vix_mod.download_vix(cache_path, force_download=force_download,
                                log=log)


def _daily_vix_features(vix: pd.DataFrame) -> pd.DataFrame:
    """Full-length per-VIX-trading-day feature frame (NaN-padded)."""
    vix = vix.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(vix["date"]).dt.normalize()
    closes = vix["vix"].to_numpy(dtype=float)
    closes3m = vix["vix3m"].to_numpy(dtype=float)
    n = len(vix)
    level = np.full(n, np.nan)
    change5 = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    for pos in range(_VIX_BURN_IN - 1, n):
        c0, c3m = closes[pos], closes3m[pos]
        if not (np.isfinite(c0) and c0 > 0 and np.isfinite(c3m) and c3m > 0):
            continue
        trail = closes[pos - _VIX_BURN_IN + 1:pos + 1]
        if not np.all(np.isfinite(trail)):
            continue
        level[pos] = c0 / 100.0 / np.sqrt(vix_mod.TRADING_DAYS_PER_YEAR)
        change5[pos] = c0 - closes[pos - 5]
        slope[pos] = c3m / c0 - 1.0
    return pd.DataFrame({
        "vix_date": dates,
        "vix_level": level,
        "vix_5d_change": change5,
        "vix_slope": slope,
    })


def add_vix_features(frame: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Merge the 3 frozen VIX features onto frame rows by t0, point-in-time.

    Each row's t0 maps to the latest VIX trading day <= t0. Rows whose
    t0 predates the VIX history raise (a fail-loud mapping bug); rows
    inside the burn-in get NaN features for the caller to drop loudly.
    """
    if vix["vix"].isna().any():
        raise ValueError("[harvix] NaN VIX closes in the series; refusing")
    feats = _daily_vix_features(vix)
    vix_ords = np.array(
        [d.toordinal() for d in feats["vix_date"]], dtype=np.int64)
    t0s = pd.to_datetime(frame["t0"]).dt.normalize()
    t0_ords = np.array([d.toordinal() for d in t0s], dtype=np.int64)
    pos = np.searchsorted(vix_ords, t0_ords, side="right") - 1
    if (pos < 0).any():
        bad = frame.loc[pos < 0, "t0"].min()
        raise ValueError(
            f"[harvix] t0 {bad} predates the VIX history; refusing")
    cols = {c: feats[c].to_numpy(dtype=float)[pos] for c in VIX_FEATURE_COLUMNS}
    out = frame.reset_index(drop=True).copy()
    for c, vals in cols.items():
        out[c] = vals
    return out


# Arm column sets. HAR_COLUMNS is the frozen 31-feature VOL set; the
# challenger adds exactly the 3 frozen VIX features.
HAR_COLUMNS: list[str] = list(vol_features.FEATURE_COLUMNS)
HARVIX_COLUMNS: list[str] = HAR_COLUMNS + VIX_FEATURE_COLUMNS
