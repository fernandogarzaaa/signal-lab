"""Tests for the EVENTVOL iv_proxy baseline (synthetic, no network).

Checks the beta x VIX/100/sqrt(252) math, the beta=1.0 fallback, the
VIX fallback, the positivity floor, and fail-loud NaN behavior.
"""

import numpy as np
import pandas as pd
import pytest

from signal_lab.eventvol import iv_proxy as iv_mod

N_DAYS = 400
START = "2022-01-03"


def _panel(beta=1.0, seed=7):
    dates = pd.bdate_range(START, periods=N_DAYS)
    rng = np.random.default_rng(seed)
    spy_rets = rng.normal(0.0005, 0.008, N_DAYS)
    stock_rets = beta * spy_rets + rng.normal(0, 0.002, N_DAYS)
    rows = []
    for t, rets in (("AAA", stock_rets), ("SPY", spy_rets)):
        close = 100.0 * np.exp(np.cumsum(rets))
        for d, c in zip(dates, close):
            rows.append({"ticker": t, "date": d, "open": c, "high": c,
                         "low": c, "adj_close": c, "volume": 1_000_000})
    prices = pd.DataFrame(rows)
    vix = pd.DataFrame({"date": dates,
                        "vix": np.full(N_DAYS, 20.0),
                        "vix3m": np.full(N_DAYS, 21.0)})
    return prices, vix, dates


def _rows(dates):
    return pd.DataFrame([{"ticker": "AAA", "t0": dates[300].date()},
                         {"ticker": "AAA", "t0": dates[350].date()}])


def test_iv_proxy_math_beta_one():
    prices, vix, dates = _panel(beta=1.0)
    fc, counts = iv_mod.forecast_iv_proxy(_rows(dates), prices, vix,
                                          log=lambda *a: None)
    expected = 20.0 / 100.0 / np.sqrt(252)
    assert fc[0] == pytest.approx(expected, rel=0.05)
    assert fc[1] == pytest.approx(expected, rel=0.05)
    assert counts["n_beta_fallback"] == 0
    assert counts["n_vix_fallback"] == 0
    assert np.all(fc > 0)


def test_iv_proxy_math_beta_two():
    prices, vix, dates = _panel(beta=2.0)
    fc, _ = iv_mod.forecast_iv_proxy(_rows(dates), prices, vix,
                                     log=lambda *a: None)
    expected = 2.0 * 20.0 / 100.0 / np.sqrt(252)
    assert fc[0] == pytest.approx(expected, rel=0.10)


def test_iv_proxy_beta_fallback_short_history():
    prices, vix, dates = _panel(beta=1.0)
    rows = pd.DataFrame([{"ticker": "AAA", "t0": dates[30].date()}])
    fc, counts = iv_mod.forecast_iv_proxy(rows, prices, vix,
                                          log=lambda *a: None)
    assert counts["n_beta_fallback"] == 1
    assert fc[0] == pytest.approx(20.0 / 100.0 / np.sqrt(252))


def test_iv_proxy_vix_fallback_counted():
    prices, vix, dates = _panel(beta=1.0)
    # Drop the VIX row for the event day: lookup falls back to the prior
    # trading day's close and the fallback is counted.
    vix2 = vix[vix["date"] != dates[300]].reset_index(drop=True)
    rows = pd.DataFrame([{"ticker": "AAA", "t0": dates[300].date()}])
    fc, counts = iv_mod.forecast_iv_proxy(rows, prices, vix2,
                                          log=lambda *a: None)
    assert counts["n_vix_fallback"] == 1
    assert fc[0] > 0


def test_iv_proxy_floor_positive():
    prices, vix, dates = _panel(beta=1.0)
    vix0 = vix.copy()
    vix0["vix"] = 0.01  # degenerate VIX: forecast must stay positive
    fc, _ = iv_mod.forecast_iv_proxy(_rows(dates), prices, vix0,
                                     log=lambda *a: None)
    assert np.all(fc >= iv_mod.FORECAST_FLOOR)


def test_iv_proxy_missing_market_leg_raises():
    prices, vix, dates = _panel(beta=1.0)
    prices = prices[prices["ticker"] != "SPY"]
    with pytest.raises(ValueError, match="market leg"):
        iv_mod.forecast_iv_proxy(_rows(dates), prices, vix,
                                 log=lambda *a: None)
