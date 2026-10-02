"""Tests for confmag.features: |excess_3d| target math, frame contracts."""

import numpy as np
import pandas as pd
import pytest

from signal_lab.confmag import features as feat_mod
from signal_lab.validation.periods import ConfirmationLeakError


def _panel(n_days=60, start="2022-01-03", seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for t in ["T00", "T01", "SPY"]:
        rets = rng.normal(0.0003, 0.012, n_days)
        close = 100.0 * np.exp(np.cumsum(rets))
        for i, d in enumerate(dates):
            rows.append({"ticker": t, "date": d, "open": close[i],
                         "high": close[i] * 1.001, "low": close[i] * 0.999,
                         "close": close[i], "adj_close": close[i],
                         "volume": 1_000_000.0})
    return pd.DataFrame(rows)


def test_target_math_known_values():
    prices = _panel(n_days=60, seed=3)
    frame = feat_mod.build_frame(prices)
    # Manual recomputation for T00 at the first t0.
    t00 = prices[prices["ticker"] == "T00"].sort_values("date")
    spy = prices[prices["ticker"] == "SPY"].sort_values("date")
    c = t00["adj_close"].to_numpy()
    s = spy["adj_close"].to_numpy()
    r_i = np.log(c[1:4]) - np.log(c[0:3])
    r_spy = np.log(s[1:4]) - np.log(s[0:3])
    expected = abs(float(np.sum(r_i - r_spy)))
    row = frame[(frame["ticker"] == "T00")].iloc[0]
    assert row["target"] == pytest.approx(expected, rel=1e-10)
    assert (frame["target"] >= 0).all()


def test_spy_excluded_and_row_counts():
    prices = _panel(n_days=60)
    frame = feat_mod.build_frame(prices)
    assert not (frame["ticker"] == "SPY").any()
    # 60 days, 3-day horizon -> 57 rows per ticker, 2 tickers.
    assert len(frame) == 2 * 57
    # t1 is t0 + 3 trading days on the union calendar.
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    for _, r in frame.sample(10, random_state=1).iterrows():
        i = cal.get_loc(pd.Timestamp(r["t0"]))
        assert cal[i + 3].date() == r["t1"]


def test_partial_windows_dropped():
    prices = _panel(n_days=10)
    frame = feat_mod.build_frame(prices)
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    assert frame["t0"].max() == cal[6].date()  # last full 3-day window


def test_missing_close_in_window_fails_loud():
    prices = _panel(n_days=60)
    prices.loc[(prices["ticker"] == "T00") &
               (prices["date"] == prices["date"].unique()[5]),
               "adj_close"] = np.nan
    with pytest.raises(Exception, match="missing close"):
        feat_mod.build_frame(prices)


def test_confirmation_never_buildable():
    prices = _panel(n_days=60, start="2026-08-01")
    with pytest.raises(ConfirmationLeakError):
        feat_mod.build_frame(prices)


def test_feature_columns_frozen():
    assert len(feat_mod.FEATURE_COLUMNS) == 34
    assert feat_mod.HORIZON_DAYS == 3
