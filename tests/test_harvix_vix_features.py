"""Tests for harvix.features: frozen VIX feature math, point-in-time-ness, burn-in."""

import numpy as np
import pandas as pd
import pytest

from signal_lab.harvix import features as feat_mod


def _vix_series(n_days=60, start="2022-01-03", seed=7, vix0=20.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    vix = vix0 + np.cumsum(rng.normal(0, 0.5, n_days))
    vix = np.clip(vix, 5.0, None)
    vix3m = vix * (1.0 + rng.normal(0.05, 0.01, n_days))
    return pd.DataFrame({"date": dates, "vix": vix, "vix3m": vix3m})


def _frame(tickers=("AAA",), n_days=60, start="2022-01-03"):
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for t in tickers:
        for i, d in enumerate(dates[:-5]):  # leave a 5-day label window
            rows.append({"ticker": t, "t0": d.date(), "t1": dates[i + 5].date(),
                         "target": 0.01})
    return pd.DataFrame(rows)


def test_vix_feature_math_known_values():
    vix = _vix_series()
    f = feat_mod.add_vix_features(_frame(), vix)
    row = f.iloc[40]
    t0 = pd.Timestamp(row["t0"]).normalize()
    pos = int(vix.index[vix["date"] == t0][0])
    c0 = vix["vix"].iloc[pos]
    c3m = vix["vix3m"].iloc[pos]
    assert row["vix_level"] == pytest.approx(c0 / 100.0 / np.sqrt(252), rel=1e-12)
    assert row["vix_5d_change"] == pytest.approx(c0 - vix["vix"].iloc[pos - 5], rel=1e-12)
    assert row["vix_slope"] == pytest.approx(c3m / c0 - 1.0, rel=1e-12)


def test_point_in_time_truncation():
    """Features for t0 <= cut are identical on the full vs truncated VIX series."""
    vix = _vix_series(n_days=120)
    frame = _frame(n_days=120)
    full = feat_mod.add_vix_features(frame, vix)
    cut = pd.Timestamp("2022-04-01")
    trunc = feat_mod.add_vix_features(frame, vix[vix["date"] <= cut].reset_index(drop=True))
    merged = full.merge(trunc, on=["ticker", "t0"], suffixes=("_full", "_tr"))
    both = merged[merged["t0"].apply(lambda d: pd.Timestamp(d) <= cut - pd.Timedelta(days=5))]
    assert len(both) > 30
    for c in feat_mod.VIX_FEATURE_COLUMNS:
        a = both[f"{c}_full"].to_numpy()
        b = both[f"{c}_tr"].to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        assert both_nan.sum() + np.isclose(
            a[~both_nan], b[~both_nan], rtol=1e-12, atol=1e-15).sum() == len(a), c


def test_burn_in_yields_nan():
    vix = _vix_series()
    f = feat_mod.add_vix_features(_frame(), vix)
    early = f.iloc[:21]
    assert early[feat_mod.VIX_FEATURE_COLUMNS].isna().all().all()
    late = f.iloc[21:]
    assert late[feat_mod.VIX_FEATURE_COLUMNS].notna().all().all()


def test_t0_before_vix_history_raises():
    vix = _vix_series()
    frame = _frame(start="2021-01-04")
    with pytest.raises(ValueError, match="predates the VIX history"):
        feat_mod.add_vix_features(frame, vix)


def test_nan_vix_series_refused():
    vix = _vix_series()
    vix.loc[30, "vix"] = np.nan
    with pytest.raises(ValueError, match="NaN VIX closes"):
        feat_mod.add_vix_features(_frame(), vix)


def test_nonpositive_close_yields_nan_not_filled():
    vix = _vix_series()
    vix.loc[40, "vix"] = -1.0
    f = feat_mod.add_vix_features(_frame(), vix)
    row = f[f["t0"] == vix["date"].iloc[40].date()]
    assert len(row) == 1
    assert row[feat_mod.VIX_FEATURE_COLUMNS].isna().all(axis=None)


def test_nan_vix3m_yields_nan_features_not_filled():
    """A missing vix3m close at t0 NaNs the row's VIX features (dropped by
    the caller per the frozen spec: rows with non-positive/missing VIX or
    VIX3M closes at t0 are dropped, never filled)."""
    vix = _vix_series()
    vix.loc[40, "vix3m"] = np.nan
    f = feat_mod.add_vix_features(_frame(), vix)
    row = f[f["t0"] == vix["date"].iloc[40].date()]
    assert len(row) == 1
    assert row[feat_mod.VIX_FEATURE_COLUMNS].isna().all(axis=None)


def test_t0_maps_to_latest_vix_day_not_beyond():
    """A t0 on a VIX gap maps to the latest prior VIX trading day."""
    vix = _vix_series()
    gap_date = vix["date"].iloc[45]
    vix = vix[vix["date"] != gap_date].reset_index(drop=True)
    frame = _frame()
    f = feat_mod.add_vix_features(frame, vix)
    row = f[f["t0"] == gap_date.date()].iloc[0]
    c0 = vix.set_index("date")["vix"][gap_date - pd.offsets.BDay(1)]
    assert row["vix_level"] == pytest.approx(c0 / 100.0 / np.sqrt(252), rel=1e-12)


def test_arm_column_sets():
    assert len(feat_mod.HAR_COLUMNS) == 31
    assert feat_mod.HARVIX_COLUMNS == feat_mod.HAR_COLUMNS + feat_mod.VIX_FEATURE_COLUMNS
    assert len(feat_mod.HARVIX_COLUMNS) == 34
    assert feat_mod.VIX_FEATURE_COLUMNS == ["vix_level", "vix_5d_change", "vix_slope"]
