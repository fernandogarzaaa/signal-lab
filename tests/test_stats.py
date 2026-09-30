"""Unit test for the event study on synthetic data with a known effect."""

import numpy as np
import pandas as pd

from signal_lab.stats import event_study


def _prices(n_days=120, tickers=("AAA",), jump=None, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_days, freq="B").date
    rows = []
    for t in tickers:
        rets = rng.normal(0, 0.01, n_days)
        if jump and t == jump[0]:
            rets[jump[1]] += jump[2]  # inject a known abnormal return
        closes = 100 * np.cumprod(1 + rets)
        for d, c in zip(dates, closes):
            rows.append({"ticker": t, "date": d, "close": c})
    # flat benchmark
    for d in dates:
        rows.append({"ticker": "SPY", "date": d, "close": 400.0})
    return pd.DataFrame(rows), dates


def test_event_study_detects_injected_effect():
    prices, dates = _prices(jump=("AAA", 60, 0.08))
    events = pd.DataFrame([{"ticker": "AAA", "event_date": dates[60], "sentiment": 0.9}])
    res = event_study(events, prices)
    row = res[res.window == "[-1,+1]"].iloc[0]
    assert row["n"] == 1
    assert row["mean_car"] > 0.05, f"failed to detect injected jump: {row['mean_car']}"


def test_event_study_null_when_no_effect():
    prices, dates = _prices(seed=123)
    events = pd.DataFrame([
        {"ticker": "AAA", "event_date": d, "sentiment": 0.9} for d in dates[10:40:5]
    ])
    res = event_study(events, prices)
    row = res[res.window == "[-1,+1]"].iloc[0]
    assert row["p_value"] > 0.05, f"false positive on pure noise: p={row['p_value']}"
