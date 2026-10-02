"""Tests for dispvol.features: frozen dispersion feature math, point-in-time-ness, burn-in."""

import numpy as np
import pandas as pd
import pytest

from signal_lab.dispvol import features as feat_mod

N_TICKERS = 60  # >= _MIN_TICKERS so the synthetic panel prints dispersion


def _panel(n_days=120, start="2022-01-03", seed=7, tickers=None):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    tickers = tickers or [f"T{i:02d}" for i in range(N_TICKERS - 1)] + ["SPY"]
    rows = []
    for t in tickers:
        rets = rng.normal(0.0003, 0.012, n_days)
        close = 100.0 * np.exp(np.cumsum(rets))
        vol = rng.integers(500_000, 2_000_000, n_days).astype(float)
        for i, d in enumerate(dates):
            rows.append({
                "ticker": t, "date": d,
                "open": close[i] * 0.999,
                "high": close[i] * (1.0 + abs(rets[i]) / 2 + 0.001),
                "low": close[i] * (1.0 - abs(rets[i]) / 2 - 0.001),
                "close": close[i], "adj_close": close[i], "volume": vol[i],
            })
    return pd.DataFrame(rows)


def _frame(prices, tickers=("T00",)):
    """One frame row per (ticker, trading day) with rv_1d attached."""
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    rng = np.random.default_rng(3)
    rows = []
    for t in tickers:
        for i, d in enumerate(cal[:-5]):  # leave a 5-day label window
            rows.append({"ticker": t, "t0": d.date(), "t1": cal[i + 5].date(),
                         "target": 0.01, "rv_1d": float(rng.uniform(1e-5, 1e-3))})
    return pd.DataFrame(rows)


def test_column_sets_frozen():
    assert feat_mod.DISPERSION_COLUMNS == [
        "disp_level", "disp_5d_change", "disp_z60",
        "disp_level_x_rv1d", "disp_5d_change_x_rv1d", "disp_z60_x_rv1d",
    ]
    assert len(feat_mod.BASELINE_COLUMNS) == 34
    assert len(feat_mod.CHALLENGER_COLUMNS) == 40
    assert set(feat_mod.CHALLENGER_COLUMNS) == (
        set(feat_mod.BASELINE_COLUMNS) | set(feat_mod.DISPERSION_COLUMNS))


def test_dispersion_math_known_values():
    prices = _panel(n_days=100)
    series = feat_mod.dispersion_series(prices)
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    day = cal[50]
    day_rets = []
    for t, grp in prices[prices["date"] <= day].groupby("ticker"):
        if t == "SPY":
            continue
        g = grp.sort_values("date")
        c = g[g["date"] <= day]["adj_close"].to_numpy()
        day_rets.append(np.log(c[-1] / c[-2]))
    expected = float(np.std(day_rets, ddof=1))
    got = float(series.loc[series["disp_date"] == day, "disp"].iloc[0])
    assert got == pytest.approx(expected, rel=1e-10)


def test_spy_excluded_from_dispersion():
    prices = _panel(n_days=100)
    before = feat_mod.dispersion_series(prices)["disp"].to_numpy()
    rigged = prices.copy()
    rigged.loc[rigged["ticker"] == "SPY", "adj_close"] *= 3.0
    after = feat_mod.dispersion_series(rigged)["disp"].to_numpy()
    assert np.array_equal(np.isnan(before), np.isnan(after))
    mask = np.isfinite(before)
    assert np.allclose(before[mask], after[mask], rtol=1e-12)


def test_too_few_tickers_prints_nan():
    prices = _panel(n_days=80, tickers=["A", "B", "SPY"])
    series = feat_mod.dispersion_series(prices)
    assert series["disp"].isna().all()


def test_point_in_time_truncation():
    """Features for t0 <= cut are identical on the full vs truncated panel."""
    prices = _panel(n_days=120)
    frame = _frame(prices)
    full = feat_mod.add_dispersion_features(frame, prices)
    cut = pd.Timestamp("2022-04-01")
    trunc_prices = prices[prices["date"] <= cut].reset_index(drop=True)
    trunc_frame = frame[frame["t0"].apply(lambda d: pd.Timestamp(d) <= cut)]
    trunc = feat_mod.add_dispersion_features(trunc_frame, trunc_prices)
    merged = full.merge(trunc, on=["ticker", "t0"], suffixes=("_full", "_tr"))
    both = merged[merged["t0"].apply(lambda d: pd.Timestamp(d) <= cut)]
    assert len(both) > 20
    for c in feat_mod.DISPERSION_COLUMNS:
        a = both[f"{c}_full"].to_numpy()
        b = both[f"{c}_tr"].to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        assert both_nan.sum() + np.isclose(
            a[~both_nan], b[~both_nan], rtol=1e-12, atol=1e-15).sum() == len(a), c


def test_no_future_cross_sectional_info():
    """Scrambling closes AFTER t0 cannot change that t0's features."""
    prices = _panel(n_days=120)
    frame = _frame(prices)
    cut = pd.Timestamp("2022-04-01")
    before = feat_mod.add_dispersion_features(frame, prices)
    rigged = prices.copy()
    future = rigged["date"] > cut
    rigged.loc[future, "adj_close"] = rigged.loc[future, "adj_close"] * 1e6 + 7.0
    after = feat_mod.add_dispersion_features(frame, rigged)
    early = before["t0"].apply(lambda d: pd.Timestamp(d) <= cut - pd.Timedelta(days=1))
    for c in feat_mod.DISPERSION_COLUMNS:
        a = before.loc[early, c].to_numpy()
        b = after.loc[early, c].to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        assert both_nan.sum() + np.isclose(
            a[~both_nan], b[~both_nan], rtol=1e-12, atol=1e-15).sum() == len(a), c


def test_burn_in_yields_nan_then_values():
    prices = _panel(n_days=120)
    frame = _frame(prices)
    f = feat_mod.add_dispersion_features(frame, prices)
    assert f.loc[:63, "disp_level"].isna().all()
    assert np.isnan(f.loc[63, "disp_level"])  # pos 63 < 64: still burn-in
    row = f.loc[64]
    for c in feat_mod.DISPERSION_COLUMNS:
        assert np.isfinite(row[c]), c


def test_interactions_use_own_rv1d():
    prices = _panel(n_days=120)
    frame = _frame(prices)
    f = feat_mod.add_dispersion_features(frame, prices)
    valid = f["disp_level"].notna()
    assert valid.sum() > 0
    for base in feat_mod.DISPERSION_BASE_COLUMNS:
        lhs = f.loc[valid, f"{base}_x_rv1d"].to_numpy()
        rhs = (f.loc[valid, base] * f.loc[valid, "rv_1d"]).to_numpy()
        assert np.allclose(lhs, rhs, rtol=1e-12)


def test_missing_rv1d_fails_loud():
    prices = _panel(n_days=120)
    frame = _frame(prices).drop(columns=["rv_1d"])
    with pytest.raises(ValueError, match="rv_1d"):
        feat_mod.add_dispersion_features(frame, prices)


def test_non_trading_day_t0_fails_loud():
    prices = _panel(n_days=120)
    frame = _frame(prices)
    frame.loc[0, "t0"] = pd.Timestamp("2022-01-08").date()  # a Saturday
    with pytest.raises(ValueError, match="not a price-panel trading day"):
        feat_mod.add_dispersion_features(frame, prices)


def test_dispersion_zscore_math():
    prices = _panel(n_days=120, seed=11)
    frame = _frame(prices)
    f = feat_mod.add_dispersion_features(frame, prices)
    series = feat_mod.dispersion_series(prices)["disp"].to_numpy()
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    row = f.loc[80]
    pos = int(np.searchsorted(cal.values.astype("datetime64[D]"),
                              np.datetime64(pd.Timestamp(row["t0"]).date())))
    window = series[pos - 59:pos + 1]
    expected = (series[pos] - window.mean()) / window.std(ddof=1)
    assert row["disp_z60"] == pytest.approx(expected, rel=1e-10)
    assert row["disp_5d_change"] == pytest.approx(
        series[pos] - series[pos - 5], rel=1e-10)
