"""Tests for the TAILQ event frame (synthetic, no network).

Covers: actuals-only filter, t0 weekend mapping, dedup, the 3-day
event-return target math, partial-window drops, and the
confirmation-period enforcement.
"""

import numpy as np
import pandas as pd
import pytest

from signal_lab.tailq import frame as frame_mod
from signal_lab.validation.periods import ConfirmationLeakError

N_DAYS = 80
START = "2023-01-02"


def _synthetic_prices(tickers=("AAA", "BBB"), start=START, n_days=N_DAYS,
                      seed=7):
    dates = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(seed)
    rows = []
    for t in tickers:
        rets = rng.normal(0.0005, 0.01, n_days)
        close = 100.0 * np.exp(np.cumsum(rets))
        for d, c in zip(dates, close):
            rows.append({"ticker": t, "date": d, "open": c, "high": c * 1.01,
                         "low": c * 0.99, "adj_close": c,
                         "volume": 1_000_000})
    return pd.DataFrame(rows)


def _earnings_row(ticker, ts, estimate=1.0, reported=1.1, surprise=10.0):
    return {"ticker": ticker, "earnings_ts": pd.Timestamp(ts),
            "eps_estimate": estimate, "reported_eps": reported,
            "surprise_pct": surprise}


def test_target_is_3day_event_return():
    prices = _synthetic_prices()
    d = pd.bdate_range(START, periods=N_DAYS)[20].date()
    frame = frame_mod.build_tailq_frame(
        pd.DataFrame([_earnings_row("AAA", pd.Timestamp(d))]), prices,
        log=lambda *a: None)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["t0"] == d
    g = prices[prices["ticker"] == "AAA"].sort_values("date").reset_index(drop=True)
    pos = int((pd.to_datetime(g["date"]).dt.date == d).argmax())
    closes = g["adj_close"].to_numpy(dtype=float)
    want = float(np.sum(np.log(closes[pos + 1:pos + 4] / closes[pos:pos + 3])))
    assert row["target"] == pytest.approx(want)
    assert row["t1"] == pd.to_datetime(g["date"]).dt.date.iloc[pos + 3]


def test_actuals_only_and_dedup():
    prices = _synthetic_prices()
    d = pd.bdate_range(START, periods=N_DAYS)[10].date()
    rows = [
        _earnings_row("AAA", pd.Timestamp(d) + pd.Timedelta(hours=16)),
        _earnings_row("AAA", pd.Timestamp(d) + pd.Timedelta(hours=18),
                      reported=1.2, surprise=20.0),  # same date: latest wins
        {"ticker": "AAA",
         "earnings_ts": pd.Timestamp(d) + pd.Timedelta(days=90, hours=16),
         "eps_estimate": 1.2, "reported_eps": None, "surprise_pct": None},
    ]
    frame = frame_mod.build_tailq_frame(pd.DataFrame(rows), prices,
                                       log=lambda *a: None)
    assert len(frame) == 1
    assert frame.iloc[0]["surprise_pct"] == 20.0


def test_t0_weekend_maps_to_next_trading_day():
    prices = _synthetic_prices()
    sat = pd.Timestamp("2023-01-07 16:00")  # Saturday
    frame = frame_mod.build_tailq_frame(
        pd.DataFrame([_earnings_row("AAA", sat)]), prices,
        log=lambda *a: None)
    assert frame.iloc[0]["t0"] == pd.Timestamp("2023-01-09").date()


def test_partial_window_dropped_raises_when_empty():
    prices = _synthetic_prices()
    last = pd.bdate_range(START, periods=N_DAYS)[-2].date()
    with pytest.raises(Exception):
        frame_mod.build_tailq_frame(
            pd.DataFrame([_earnings_row("AAA", pd.Timestamp(last))]), prices,
            log=lambda *a: None)


def test_confirmation_enforcement():
    prices = _synthetic_prices(start="2026-06-01", n_days=120)
    # An event in the confirmation period must raise, never be labeled.
    d = pd.Timestamp("2026-09-10").date()
    with pytest.raises(ConfirmationLeakError):
        frame_mod.build_tailq_frame(
            pd.DataFrame([_earnings_row("AAA", pd.Timestamp(d))]), prices,
            log=lambda *a: None)


def test_dev_frame_excludes_test_and_confirmation():
    prices = _synthetic_prices(start="2026-01-01", n_days=200)
    d1 = pd.Timestamp("2026-03-10").date()
    d2 = pd.Timestamp("2026-07-15").date()
    frame = frame_mod.build_tailq_frame(
        pd.DataFrame([_earnings_row("AAA", pd.Timestamp(d1)),
                      _earnings_row("BBB", pd.Timestamp(d2))]), prices,
        log=lambda *a: None)
    dev = frame_mod.dev_frame(frame)
    assert len(dev) == 1
    assert dev.iloc[0]["t0"] == d1
    test = frame_mod.test_frame(frame)
    assert len(test) == 1
    assert test.iloc[0]["t0"] == d2
