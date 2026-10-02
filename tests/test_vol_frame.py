"""Tests for vol.frame: target math, partial-window drops, confirmation freeze.

All panels are synthetic; no network.
"""

import numpy as np
import pandas as pd
import pytest

from signal_lab.validation.periods import ConfirmationLeakError
from signal_lab.vol import frame as frame_mod
from signal_lab.vol.prices import DataQualityError


def _synthetic_prices(tickers=("AAA", "BBB"), n_days=40,
                      start="2022-01-03", seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for t in tickers:
        rets = rng.normal(0.0005, 0.01, n_days)
        close = 100.0 * np.exp(np.cumsum(rets))
        for i, d in enumerate(dates):
            rows.append({
                "ticker": t, "date": d,
                "open": close[i] * 0.999, "high": close[i] * 1.002,
                "low": close[i] * 0.998, "close": close[i],
                "adj_close": close[i], "volume": 1_000_000,
            })
    return pd.DataFrame(rows)


def test_target_math_vs_hand_computation():
    prices = _synthetic_prices()
    f = frame_mod.build_frame(prices)
    # hand-compute the target for (AAA, t0 = 11th trading day)
    aaa = prices[prices["ticker"] == "AAA"].sort_values("date").reset_index(drop=True)
    closes = aaa["adj_close"].to_numpy()
    logc = np.log(closes)
    r = np.diff(logc)  # r[i] is the return of day i+1 vs day i
    t0_idx = 10
    fwd = r[t0_idx:t0_idx + 5]  # returns of days t0+1 .. t0+5
    expected = float(np.std(fwd, ddof=1))
    t0 = aaa.loc[t0_idx, "date"].date()
    got = f[(f["ticker"] == "AAA") & (f["t0"] == t0)]["target"].iloc[0]
    assert got == pytest.approx(expected, rel=1e-12)


def test_partial_windows_dropped():
    prices = _synthetic_prices(n_days=40)
    f = frame_mod.build_frame(prices)
    # 40 days -> t0 indices 0..34 per ticker
    assert len(f) == 2 * 35
    cal = frame_mod.trading_calendar(prices)
    assert f["t0"].max() == cal[34].date()
    # t1 is always t0 + 5 trading days
    assert (f["t1"] == f["t0"].apply(
        lambda d: cal[cal.get_loc(pd.Timestamp(d)) + 5].date())).all()


def test_published_at_is_t0_utc():
    prices = _synthetic_prices()
    f = frame_mod.build_frame(prices)
    assert (pd.to_datetime(f["published_at"], utc=True).dt.date == f["t0"]).all()


def test_confirmation_rows_rejected():
    prices = _synthetic_prices(start="2026-09-01", n_days=40)
    with pytest.raises(ConfirmationLeakError):
        frame_mod.build_frame(prices)


def test_missing_close_in_label_window_raises():
    prices = _synthetic_prices()
    mask = (prices["ticker"] == "AAA") & (prices["date"] == prices["date"].unique()[12])
    prices.loc[mask, "adj_close"] = np.nan
    with pytest.raises(DataQualityError):
        frame_mod.build_frame(prices)


def test_dev_and_test_filters():
    # 110 business days from 2026-03-02 span dev and test, never confirmation
    prices = _synthetic_prices(start="2026-03-02", n_days=110)
    f = frame_mod.build_frame(prices)
    dev = frame_mod.dev_frame(f)
    assert (dev["t0"] <= pd.Timestamp("2026-06-30").date()).all()
    assert len(dev) > 0
    test = frame_mod.test_frame(f)
    assert ((test["t0"] >= pd.Timestamp("2026-07-01").date())
            & (test["t0"] <= pd.Timestamp("2026-08-31").date())).all()
