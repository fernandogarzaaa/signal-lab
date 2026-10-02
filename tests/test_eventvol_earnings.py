"""Tests for the EVENTVOL earnings event frame (synthetic, no network).

Covers: actuals-only filter, t0 mapping (non-trading day -> next
trading day), dedup (latest timestamp wins), frozen target labeling,
partial-window drops, and the confirmation-period enforcement.
"""

import numpy as np
import pandas as pd
import pytest

from signal_lab.eventvol import earnings as earn_mod
from signal_lab.validation.periods import ConfirmationLeakError

N_DAYS = 60
START = "2023-01-02"


def _synthetic_prices(tickers=("AAA", "BBB")):
    dates = pd.bdate_range(START, periods=N_DAYS)
    rng = np.random.default_rng(7)
    rows = []
    for t in tickers:
        rets = rng.normal(0.0005, 0.01, N_DAYS)
        close = 100.0 * np.exp(np.cumsum(rets))
        for d, c in zip(dates, close):
            rows.append({"ticker": t, "date": d, "open": c, "high": c * 1.01,
                         "low": c * 0.99, "adj_close": c, "volume": 1_000_000})
    return pd.DataFrame(rows)


def _earnings_row(ticker, ts, estimate=1.0, reported=1.1, surprise=10.0):
    return {"ticker": ticker, "earnings_ts": pd.Timestamp(ts),
            "eps_estimate": estimate, "reported_eps": reported,
            "surprise_pct": surprise}


def test_actuals_only_filter():
    prices = _synthetic_prices()
    d = pd.bdate_range(START, periods=N_DAYS)[10].date()
    rows = [
        _earnings_row("AAA", pd.Timestamp(d) + pd.Timedelta(hours=16)),
        # estimate-only row: no reported EPS -> excluded
        {"ticker": "AAA",
         "earnings_ts": pd.Timestamp(d) + pd.Timedelta(days=90, hours=16),
         "eps_estimate": 1.2, "reported_eps": None, "surprise_pct": None},
    ]
    frame = earn_mod.build_event_frame(pd.DataFrame(rows), prices,
                                       log=lambda *a: None)
    assert len(frame) == 1
    assert frame.iloc[0]["t0"] == d


def test_t0_weekend_maps_to_next_trading_day():
    prices = _synthetic_prices()
    # 2023-01-07 is a Saturday.
    sat = pd.Timestamp("2023-01-07 16:00")
    frame = earn_mod.build_event_frame(
        pd.DataFrame([_earnings_row("AAA", sat)]), prices,
        log=lambda *a: None)
    assert len(frame) == 1
    # Next trading day after Saturday 2023-01-07 is Monday 2023-01-09.
    assert frame.iloc[0]["t0"] == pd.Timestamp("2023-01-09").date()


def test_dedup_latest_timestamp_wins():
    prices = _synthetic_prices()
    d = pd.bdate_range(START, periods=N_DAYS)[10]
    rows = [
        _earnings_row("AAA", d + pd.Timedelta(hours=16), reported=1.1,
                      surprise=10.0),
        _earnings_row("AAA", d + pd.Timedelta(hours=18), reported=1.2,
                      surprise=20.0),
    ]
    frame = earn_mod.build_event_frame(pd.DataFrame(rows), prices,
                                       log=lambda *a: None)
    assert len(frame) == 1
    assert frame.iloc[0]["reported_eps"] == 1.2
    assert frame.iloc[0]["surprise_pct"] == 20.0


def test_target_matches_manual_realized_vol():
    prices = _synthetic_prices(tickers=("AAA",))
    d = pd.bdate_range(START, periods=N_DAYS)[10]
    frame = earn_mod.build_event_frame(
        pd.DataFrame([_earnings_row("AAA", d + pd.Timedelta(hours=16))]),
        prices, log=lambda *a: None)
    g = prices.sort_values("date").reset_index(drop=True)
    pos = int((pd.to_datetime(g["date"]).dt.date == d.date()).argmax())
    window = g["adj_close"].to_numpy(dtype=float)[pos:pos + 6]
    fwd = np.log(window[1:]) - np.log(window[:-1])
    expected = float(np.std(fwd, ddof=1))
    assert frame.iloc[0]["target"] == pytest.approx(expected)


def test_partial_label_window_dropped():
    prices = _synthetic_prices()
    # Event 3 trading days before the panel end: window is partial.
    d = pd.bdate_range(START, periods=N_DAYS)[-3]
    # Partial windows are dropped; a frame with zero surviving rows
    # raises DataQualityError (fail loud, never an empty frame).
    with pytest.raises(Exception):
        earn_mod.build_event_frame(
            pd.DataFrame([_earnings_row("AAA", d + pd.Timedelta(hours=16))]),
            prices, log=lambda *a: None)


def test_prev_surprise_is_previous_quarter():
    prices = _synthetic_prices()
    dates = pd.bdate_range(START, periods=N_DAYS)
    rows = [
        _earnings_row("AAA", dates[10] + pd.Timedelta(hours=16),
                      surprise=10.0),
        _earnings_row("AAA", dates[40] + pd.Timedelta(hours=16),
                      surprise=30.0),
    ]
    frame = earn_mod.build_event_frame(pd.DataFrame(rows), prices,
                                       log=lambda *a: None)
    assert len(frame) == 2
    first = frame.iloc[0]
    second = frame.iloc[1]
    assert pd.isna(first["prev_surprise_pct"])
    assert second["prev_surprise_pct"] == pytest.approx(0.10)


def test_confirmation_event_raises():
    dates = pd.bdate_range("2026-08-01", periods=40)
    rng = np.random.default_rng(7)
    rows = []
    for d, c in zip(dates, 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 40)))):
        rows.append({"ticker": "AAA", "date": d, "open": c, "high": c,
                     "low": c, "adj_close": c, "volume": 1_000_000})
    prices = pd.DataFrame(rows)
    # dates[30] ~= 2026-09-11: inside the frozen confirmation period.
    ev = pd.DataFrame([_earnings_row("AAA", dates[30] + pd.Timedelta(hours=16))])
    with pytest.raises(ConfirmationLeakError):
        earn_mod.build_event_frame(ev, prices, log=lambda *a: None)


def test_dev_and_test_frame_periods():
    prices = _synthetic_prices()
    d = pd.bdate_range(START, periods=N_DAYS)[10].date()
    frame = earn_mod.build_event_frame(
        pd.DataFrame([_earnings_row(
            "AAA", pd.Timestamp(d) + pd.Timedelta(hours=16))]),
        prices, log=lambda *a: None)
    dev = earn_mod.dev_frame(frame)
    assert len(dev) == 1  # 2023 is in development
    test = earn_mod.test_frame(frame)
    assert len(test) == 0
    bad = frame.copy()
    bad.loc[0, "t0"] = pd.Timestamp("2026-09-15").date()
    with pytest.raises(ConfirmationLeakError):
        earn_mod.dev_frame(bad)


def test_event_past_calendar_end_dropped_not_raised():
    prices = _synthetic_prices()
    last_trading = pd.bdate_range(START, periods=N_DAYS)[-1]
    d_inside = pd.bdate_range(START, periods=N_DAYS)[10]
    d_past = last_trading + pd.offsets.BDay(3)
    rows = [
        _earnings_row("AAA", d_inside + pd.Timedelta(hours=16)),
        _earnings_row("AAA", d_past + pd.Timedelta(hours=16)),
    ]
    logged = []
    frame = earn_mod.build_event_frame(pd.DataFrame(rows), prices,
                                       log=logged.append)
    assert len(frame) == 1
    assert frame.iloc[0]["t0"] == d_inside.date()
    assert any("past the price calendar end" in m for m in logged)
