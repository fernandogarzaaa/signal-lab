"""Tests for TAILQ features: point-in-time safety and frozen spec (synthetic).

Covers: all 15 frozen feature columns are produced, feature values at
t0 are invariant to post-t0 prices (point-in-time), first-event
prev_surprise fills 0.0, beta falls back to 1.0 on insufficient
history, rows with insufficient history are dropped, and the
confirmation enforcement.
"""

import numpy as np
import pandas as pd
import pytest

from signal_lab.tailq import features as feat_mod
from signal_lab.validation.periods import ConfirmationLeakError


def _synthetic_prices(tickers=("AAA", "SPY"), n_days=300, start="2023-01-02",
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


def _synthetic_vix(n_days=300, start="2023-01-02", seed=7):
    dates = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(seed)
    vix = 15.0 + np.abs(rng.normal(0, 2, n_days))
    vix3m = vix * (1.0 + rng.normal(0.05, 0.02, n_days))
    return pd.DataFrame({"date": dates, "vix": vix, "vix3m": vix3m})


def _events(t0_dates, ticker="AAA"):
    rows = []
    for i, d in enumerate(t0_dates):
        rows.append({
            "ticker": ticker, "t0": pd.Timestamp(d).date(),
            "t1": (pd.Timestamp(d) + pd.offsets.BDay(3)).date(),
            "published_at": pd.Timestamp(d, tz="UTC"),
            "target": -0.02, "eps_estimate": 1.0, "reported_eps": 1.1,
            "surprise_pct": 10.0 if i else 5.0,
            "prev_surprise_pct": np.nan if i == 0 else 0.05,
        })
    return pd.DataFrame(rows)


def test_all_fifteen_features_present():
    prices = _synthetic_prices()
    vix = _synthetic_vix()
    d = pd.bdate_range("2023-01-02", periods=300)[200].date()
    feats = feat_mod.build_features(_events([d]), prices, vix,
                                    log=lambda *a: None)
    assert len(feats) == 1
    for c in feat_mod.FEATURE_COLUMNS:
        assert c in feats.columns, c
        assert np.isfinite(feats.iloc[0][c]), c


def test_point_in_time_invariant_to_future_prices():
    prices = _synthetic_prices()
    vix = _synthetic_vix()
    d = pd.bdate_range("2023-01-02", periods=300)[200].date()
    base = feat_mod.build_features(_events([d]), prices, vix,
                                   log=lambda *a: None)
    # Corrupt all prices after t0 (plus 10x shock): features must not move.
    shocked = prices.copy()
    mask = (shocked["ticker"] == "AAA") & (
        pd.to_datetime(shocked["date"]).dt.date > d)
    shocked.loc[mask, "adj_close"] *= 10.0
    again = feat_mod.build_features(_events([d]), shocked, vix,
                                    log=lambda *a: None)
    for c in feat_mod.FEATURE_COLUMNS:
        assert base.iloc[0][c] == pytest.approx(again.iloc[0][c]), c


def test_rv_math_matches_hand_computation():
    prices = _synthetic_prices()
    vix = _synthetic_vix()
    d = pd.bdate_range("2023-01-02", periods=300)[200].date()
    feats = feat_mod.build_features(_events([d]), prices, vix,
                                    log=lambda *a: None)
    g = prices[prices["ticker"] == "AAA"].sort_values("date").reset_index(drop=True)
    closes = g["adj_close"].to_numpy(dtype=float)
    rets = np.log(closes[1:] / closes[:-1])
    pos_r = int((pd.to_datetime(g["date"]).dt.date == d).argmax()) - 1
    # rets[j] is the return on closes date index j + 1.
    assert feats.iloc[0]["rv_5d"] == pytest.approx(
        float(np.std(rets[pos_r - 4:pos_r + 1], ddof=1)))
    assert feats.iloc[0]["ret_3d_pre"] == pytest.approx(
        float(np.sum(rets[pos_r - 2:pos_r + 1])))


def test_prev_surprise_first_event_fills_zero():
    prices = _synthetic_prices()
    vix = _synthetic_vix()
    d = pd.bdate_range("2023-01-02", periods=300)[200].date()
    feats = feat_mod.build_features(_events([d]), prices, vix,
                                    log=lambda *a: None)
    assert feats.iloc[0]["prev_surprise"] == 0.0


def test_beta_fallback_on_short_history():
    prices = _synthetic_prices(n_days=100)
    vix = _synthetic_vix(n_days=100)
    d = pd.bdate_range("2023-01-02", periods=100)[90].date()
    feats = feat_mod.build_features(_events([d]), prices, vix,
                                    log=lambda *a: None)
    # 90 paired obs >= 60 min: beta is estimated, finite.
    assert np.isfinite(feats.iloc[0]["beta_252"])
    # Too-short history: NaN feature rows are dropped (66-day leg).
    d_early = pd.bdate_range("2023-01-02", periods=100)[40].date()
    with pytest.raises(ValueError, match="zero rows"):
        feat_mod.build_features(_events([d_early]), prices, vix,
                                log=lambda *a: None)


def test_confirmation_enforcement():
    prices = _synthetic_prices(start="2026-01-01", n_days=300)
    vix = _synthetic_vix(start="2026-01-01", n_days=300)
    d = pd.Timestamp("2026-09-10").date()
    with pytest.raises(ConfirmationLeakError):
        feat_mod.build_features(_events([d]), prices, vix,
                                log=lambda *a: None)


def test_spy_panel_required():
    prices = _synthetic_prices(tickers=("AAA",))
    vix = _synthetic_vix()
    d = pd.bdate_range("2023-01-02", periods=300)[200].date()
    with pytest.raises(ValueError, match="SPY"):
        feat_mod.build_features(_events([d]), prices, vix,
                                log=lambda *a: None)
