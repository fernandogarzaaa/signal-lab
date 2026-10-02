"""Tests for vol.features: point-in-time-ness, known values, NaN burn-in."""

import numpy as np
import pandas as pd
import pytest

from signal_lab.vol import features as feat_mod
from signal_lab.vol import frame as frame_mod


def _panel(tickers=("AAA", "SPY"), n_days=60, start="2022-01-03", seed=21):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
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


def test_point_in_time_truncation():
    """Features for t0 <= cut are identical on the full vs truncated panel."""
    prices = _panel()
    full = feat_mod.add_features(frame_mod.build_frame(prices), prices)
    cut = pd.Timestamp("2022-03-01")
    trunc_prices = prices[prices["date"] <= cut].reset_index(drop=True)
    trunc = feat_mod.add_features(frame_mod.build_frame(trunc_prices), trunc_prices)
    # truncated frame only reaches cut - 5 trading days (partial dropped)
    merged = full.merge(trunc, on=["ticker", "t0"], suffixes=("_full", "_tr"))
    assert len(merged) > 50
    for c in feat_mod.FEATURE_COLUMNS:
        a = merged[f"{c}_full"].to_numpy()
        b = merged[f"{c}_tr"].to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        assert both_nan.sum() + np.isclose(a[~both_nan], b[~both_nan],
                                           rtol=1e-12, atol=1e-15).sum() == len(a), c


def test_rv_1d_equals_squared_return():
    prices = _panel()
    f = feat_mod.add_features(frame_mod.build_frame(prices), prices)
    row = f[(f["ticker"] == "AAA")].iloc[30]
    aaa = prices[prices["ticker"] == "AAA"].sort_values("date").reset_index(drop=True)
    closes = aaa["adj_close"].to_numpy()
    pos = aaa.index[aaa["date"] == pd.Timestamp(row["t0"])][0]
    r_t0 = np.log(closes[pos] / closes[pos - 1])
    assert row["rv_1d"] == pytest.approx(r_t0 ** 2, rel=1e-12)


def test_parkinson_known_value():
    prices = _panel()
    f = feat_mod.add_features(frame_mod.build_frame(prices), prices)
    row = f[(f["ticker"] == "AAA")].iloc[30]
    aaa = prices[prices["ticker"] == "AAA"].sort_values("date").reset_index(drop=True)
    prow = aaa[aaa["date"] == pd.Timestamp(row["t0"])].iloc[0]
    expected = (np.log(prow["high"] / prow["low"])) ** 2 / (4.0 * np.log(2.0))
    assert row["parkinson_1d"] == pytest.approx(expected, rel=1e-12)


def test_dow_one_hot():
    prices = _panel()
    f = feat_mod.add_features(frame_mod.build_frame(prices), prices)
    for _, row in f.head(50).iterrows():
        dow = pd.Timestamp(row["t0"]).weekday()
        assert row[[f"dow_{i}" for i in range(5)]].sum() == 1.0
        assert row[f"dow_{dow}"] == 1.0


def test_nan_on_insufficient_history_never_filled():
    prices = _panel()
    f = feat_mod.add_features(frame_mod.build_frame(prices), prices)
    aaa = f[f["ticker"] == "AAA"].reset_index(drop=True)
    # first frame row: no trailing history at all
    assert np.isnan(aaa.loc[0, "rv_22d"])
    assert np.isnan(aaa.loc[0, "rv_5d"])
    # rv_22d needs 22 trailing squared returns; r[0] is NaN (no prior
    # close), so row 22 is the first with a full window
    assert np.isnan(aaa.loc[21, "rv_22d"])
    assert np.isfinite(aaa.loc[22, "rv_22d"])
    clean, n_dropped = feat_mod.drop_nan_features(f)
    assert n_dropped > 0
    assert clean[feat_mod.FEATURE_COLUMNS].notna().all().all()


def test_spy_legs_present():
    prices = _panel()
    f = feat_mod.add_features(frame_mod.build_frame(prices), prices)
    aaa = f[f["ticker"] == "AAA"].reset_index(drop=True)
    spy = f[f["ticker"] == "SPY"].reset_index(drop=True)
    # SPY legs on AAA rows equal SPY's own-leg values at the same t0
    m = aaa.merge(spy[["t0", "rv_5d"]], on="t0", suffixes=("", "_spyown"))
    both = m[np.isfinite(m["spy_rv_5d"]) & np.isfinite(m["rv_5d_spyown"])]
    assert len(both) > 0
    assert np.allclose(both["spy_rv_5d"], both["rv_5d_spyown"], rtol=1e-12)
