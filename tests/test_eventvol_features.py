"""Tests for EVENTVOL features: point-in-time safety, beta math, NaN policy.

All synthetic, no network. The point-in-time truncation test mirrors the
VOL convention: features for t0 <= cut are identical whether computed
on the full panel or on the panel truncated at cut.
"""

import numpy as np
import pandas as pd
import pytest

from signal_lab.eventvol import features as feat_mod
from signal_lab.eventvol import vix as vix_mod

N_DAYS = 120
START = "2023-01-02"


def _synthetic_prices(tickers=("AAA", "SPY"), seed=7):
    dates = pd.bdate_range(START, periods=N_DAYS)
    rng = np.random.default_rng(seed)
    rows = []
    for t in tickers:
        rets = rng.normal(0.0005, 0.01, N_DAYS)
        close = 100.0 * np.exp(np.cumsum(rets))
        for d, c in zip(dates, close):
            rows.append({"ticker": t, "date": d, "open": c, "high": c * 1.01,
                         "low": c * 0.99, "adj_close": c, "volume": 1_000_000})
    return pd.DataFrame(rows)


def _synthetic_vix():
    dates = pd.bdate_range(START, periods=N_DAYS)
    rng = np.random.default_rng(11)
    v = 18.0 + np.cumsum(rng.normal(0, 0.3, N_DAYS))
    v = np.clip(v, 10.0, 40.0)
    return pd.DataFrame({"date": dates, "vix": v, "vix3m": v * 1.08})


def _frame_at(dates):
    rows = []
    for d in dates:
        rows.append({"ticker": "AAA", "t0": d.date(),
                     "t1": (d + pd.offsets.BDay(5)).date(),
                     "published_at": pd.Timestamp(d.date(), tz="UTC"),
                     "target": 0.01, "eps_estimate": 1.0,
                     "reported_eps": 1.1, "surprise_pct": 10.0,
                     "prev_surprise_pct": 0.05})
    return pd.DataFrame(rows)


def test_point_in_time_truncation():
    prices = _synthetic_prices()
    vix = _synthetic_vix()
    dates = pd.bdate_range(START, periods=N_DAYS)
    cut = dates[79]
    t0s = [dates[40], dates[60], dates[100]]
    frame = _frame_at(t0s)
    full = feat_mod.add_features(frame, prices, vix, log=lambda *a: None)
    trunc_prices = prices[pd.to_datetime(prices["date"]) <= cut]
    trunc_vix = vix[vix["date"] <= cut]
    trunc_frame = frame[pd.to_datetime(frame["t0"]) <= cut].reset_index(drop=True)
    trunc = feat_mod.add_features(trunc_frame, trunc_prices, trunc_vix,
                                  log=lambda *a: None)
    cols = feat_mod.CHALLENGER_FEATURES
    for i in (0, 1):  # t0 <= cut only
        for c in cols:
            a, b = full.iloc[i][c], trunc.iloc[i][c]
            if pd.isna(a) and pd.isna(b):
                continue
            assert a == pytest.approx(b), (c, i)


def test_vix_features_math():
    dates = pd.bdate_range(START, periods=N_DAYS)
    vix = pd.DataFrame({"date": dates,
                        "vix": np.full(N_DAYS, 20.0),
                        "vix3m": np.full(N_DAYS, 22.0)})
    f = vix_mod.vix_features_at(dates[50].date(), vix)
    assert f["vix_level"] == pytest.approx(20.0 / 100.0 / np.sqrt(252))
    assert f["vix_5d_change"] == pytest.approx(0.0)
    assert f["vix_runup_22d"] == pytest.approx(0.0)
    assert f["vix_slope"] == pytest.approx(0.10)


def test_vix_features_insufficient_history_nan():
    dates = pd.bdate_range(START, periods=N_DAYS)
    vix = pd.DataFrame({"date": dates,
                        "vix": np.full(N_DAYS, 20.0),
                        "vix3m": np.full(N_DAYS, 22.0)})
    f = vix_mod.vix_features_at(dates[10].date(), vix)
    assert all(pd.isna(v) for v in f.values())


def test_vix_at_fallback_counted():
    dates = pd.bdate_range(START, periods=N_DAYS)
    vix = pd.DataFrame({"date": dates,
                        "vix": np.full(N_DAYS, 20.0),
                        "vix3m": np.full(N_DAYS, 22.0)})
    # Saturday: falls back to Friday's close.
    c, _, fb = vix_mod.vix_at(pd.Timestamp("2023-01-07").date(), vix)
    assert c == pytest.approx(20.0)
    assert fb is True
    c, _, fb = vix_mod.vix_at(dates[50].date(), vix)
    assert fb is False


def test_trailing_beta_two_x_spy():
    rng = np.random.default_rng(7)
    n = 300
    spy = rng.normal(0, 0.01, n)
    stock = 2.0 * spy + rng.normal(0, 0.001, n)
    beta, fb = feat_mod.trailing_beta(stock, spy, n - 1)
    assert fb is False
    assert beta == pytest.approx(2.0, abs=0.05)


def test_trailing_beta_fallback_short_history():
    rng = np.random.default_rng(7)
    spy = rng.normal(0, 0.01, 30)
    stock = rng.normal(0, 0.01, 30)
    beta, fb = feat_mod.trailing_beta(stock, spy, 29)
    assert beta == 1.0
    assert fb is True


def test_drop_nan_features_counts():
    prices = _synthetic_prices()
    vix = _synthetic_vix()
    dates = pd.bdate_range(START, periods=N_DAYS)
    # t0 at dates[10]: insufficient history -> NaN features -> dropped.
    frame = _frame_at([dates[10], dates[60]])
    feat = feat_mod.add_features(frame, prices, vix, log=lambda *a: None)
    clean, n = feat_mod.drop_nan_features(feat, log=lambda *a: None)
    assert n == 1
    assert len(clean) == 1
    assert clean.iloc[0]["t0"] == dates[60].date()


def test_feature_sets_frozen():
    assert len(feat_mod.HARVIX_FEATURES) == 31 + 4
    assert len(feat_mod.CHALLENGER_FEATURES) == 31 + 4 + 4
    assert set(feat_mod.EVENT_FEATURE_COLUMNS) == {
        "event_day_absret", "event_day_ret", "trailing_surprise", "beta_252"}
