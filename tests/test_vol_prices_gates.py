"""Tests for vol.prices fail-loud data-quality gates (synthetic panels)."""

import numpy as np
import pandas as pd
import pytest

from signal_lab.vol import prices as prices_mod
from signal_lab.vol.prices import DataQualityError


def _clean_panel(n_days=30, tickers=("AAA", "BBB"), start="2022-01-03", seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for t in tickers:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
        for i, d in enumerate(dates):
            rows.append({
                "ticker": t, "date": d,
                "open": close[i], "high": close[i] * 1.001,
                "low": close[i] * 0.999, "close": close[i],
                "adj_close": close[i],
                "volume": float(rng.integers(100_000, 1_000_000)),
            })
    return pd.DataFrame(rows)


def test_clean_panel_passes():
    out = prices_mod.validate_prices(_clean_panel(), log=lambda *a: None)
    assert len(out) == 60


def test_duplicate_ticker_date_raises():
    p = _clean_panel()
    p = pd.concat([p, p.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataQualityError, match="duplicate"):
        prices_mod.validate_prices(p, log=lambda *a: None)


def test_timestamp_ordering_violation_raises():
    p = _clean_panel()
    idx = p[p["ticker"] == "AAA"].index
    # swap two dates within AAA
    d = p.loc[idx, "date"].to_numpy().copy()
    d[3], d[4] = d[4], d[3]
    p.loc[idx, "date"] = d
    with pytest.raises(DataQualityError, match="ordering"):
        prices_mod.validate_prices(p, log=lambda *a: None)


def test_nan_close_raises():
    p = _clean_panel()
    p.loc[7, "close"] = np.nan
    with pytest.raises(DataQualityError, match="NaN closes"):
        prices_mod.validate_prices(p, log=lambda *a: None)


def test_prelisting_prefix_trimmed_not_filled():
    p = _clean_panel()
    idx = p[p["ticker"] == "AAA"].index[:10]
    p.loc[idx, "close"] = np.nan
    p.loc[idx, "adj_close"] = np.nan
    trimmed = prices_mod.trim_prelisting_history(p, log=lambda *a: None)
    assert len(trimmed) == len(p) - 10
    # AAA now starts at its first traded day
    aaa = trimmed[trimmed["ticker"] == "AAA"]
    assert aaa["close"].notna().all()
    # and the trimmed panel passes validation
    prices_mod.validate_prices(trimmed, log=lambda *a: None)


def test_all_nan_ticker_raises():
    p = _clean_panel()
    idx = p[p["ticker"] == "AAA"].index
    p.loc[idx, "close"] = np.nan
    with pytest.raises(DataQualityError, match="no valid closes"):
        prices_mod.trim_prelisting_history(p, log=lambda *a: None)


def test_zero_volume_stretch_raises():
    p = _clean_panel()
    idx = p[p["ticker"] == "BBB"].index[:3]
    p.loc[idx, "volume"] = 0.0
    with pytest.raises(DataQualityError, match="zero-volume"):
        prices_mod.validate_prices(p, log=lambda *a: None)


def test_isolated_zero_volume_days_pass():
    p = _clean_panel()
    idx = p[p["ticker"] == "BBB"].index
    p.loc[idx[3], "volume"] = 0.0
    p.loc[idx[20], "volume"] = 0.0
    out = prices_mod.validate_prices(p, log=lambda *a: None)
    assert len(out) == 60
