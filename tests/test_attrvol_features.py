"""Tests for attrvol.features: event detection, decomposition math, analogues, point-in-time-ness."""

import numpy as np
import pandas as pd
import pytest

from signal_lab.attrvol import features as feat_mod

N_TICKERS = 40


def _panel(n_days=300, start="2022-01-03", seed=7, tickers=None,
           spike=None):
    """Synthetic price panel. spike = (ticker, day_index, size) one-day shock."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    tickers = tickers or [f"T{i:02d}" for i in range(N_TICKERS - 1)] + ["SPY"]
    rows = []
    for t in tickers:
        rets = rng.normal(0.0003, 0.012, n_days)
        if spike is not None and t == spike[0]:
            rets[spike[1]] += spike[2]
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


def test_column_sets_frozen():
    assert feat_mod.ATTR_COLUMNS == [
        "driver_H", "idio_share", "ret_x_H",
        "analogue_vol_mean", "analogue_vol_std",
    ]
    assert len(feat_mod.BASELINE_COLUMNS) == 34
    assert len(feat_mod.CHALLENGER_COLUMNS) == 39
    assert set(feat_mod.CHALLENGER_COLUMNS) == (
        set(feat_mod.BASELINE_COLUMNS) | set(feat_mod.ATTR_COLUMNS))


def test_event_detection_finds_spike():
    prices = _panel(spike=("T00", 250, 0.15))
    events = feat_mod.detect_events(prices)
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    spike_day = cal[250]
    hit = events[(events["ticker"] == "T00") &
                 (events["t0"] == spike_day)]
    assert len(hit) == 1
    # A calm day for the same ticker is not an event.
    calm = events[(events["ticker"] == "T00") & (events["t0"] == cal[100])]
    assert len(calm) == 0
    # SPY itself is never an event row.
    assert not (events["ticker"] == "SPY").any()


def test_event_quantile_strictly_trailing():
    """Q90 at t0 uses only the 252 days strictly before t0."""
    prices = _panel(n_days=300, seed=11)
    events = feat_mod.detect_events(prices)
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    # Recompute manually for one ticker/day.
    px = prices[prices["ticker"] == "T05"].sort_values("date")
    spy = prices[prices["ticker"] == "SPY"].sort_values("date")
    r = np.log(px["adj_close"].to_numpy() /
               np.roll(px["adj_close"].to_numpy(), 1))[1:]
    r_spy = np.log(spy["adj_close"].to_numpy() /
                   np.roll(spy["adj_close"].to_numpy(), 1))[1:]
    abn = np.abs(r - r_spy)
    # Day index 200 in return space == cal[201] (r[0] is the cal[1] return).
    day = cal[201]
    lo = max(0, 200 - 252)
    trailing = abn[lo:200]  # strictly before cal[201]
    assert len(trailing) >= 60
    expected_q90 = float(np.quantile(trailing, 0.90))
    is_event_expected = abn[200] >= expected_q90
    hit = events[(events["ticker"] == "T05") & (events["t0"] == day)]
    assert (len(hit) == 1) == is_event_expected


def test_decompose_event_math():
    # beta=1.2, market +1%, stock +5%: m=0.012, e=0.038.
    h, s_e, rxh = feat_mod.decompose_event(1.2, 0.01, 0.05)
    denom = 0.012 + 0.038
    assert h == pytest.approx((0.012 / denom) ** 2 + (0.038 / denom) ** 2)
    assert s_e == pytest.approx(0.038 / denom)
    assert rxh == pytest.approx(0.05 * h)
    assert 0.5 <= h <= 1.0
    # Fully idiosyncratic move -> H = 1.
    h2, s2, _ = feat_mod.decompose_event(0.0, 0.01, 0.05)
    assert h2 == pytest.approx(1.0)
    assert s2 == pytest.approx(1.0)
    # Zero move -> NaN triple.
    h3, _, _ = feat_mod.decompose_event(1.0, 0.0, 0.0)
    assert np.isnan(h3)


def _repeated_pattern_panel():
    """Two identical 30-day blocks; the block ends are the analogue pair."""
    rng = np.random.default_rng(5)
    n_days, start = 320, "2022-01-03"
    dates = pd.bdate_range(start, periods=n_days)
    tickers = ["T00", "T01", "SPY"]
    pattern = rng.normal(0.001, 0.02, 30)
    pattern[-1] = 0.12  # spike on the block's last day
    rows = []
    for t in tickers:
        rets = rng.normal(0.0003, 0.012, n_days)
        if t == "T00":
            rets[200:230] = pattern
            rets[270:300] = pattern
        close = 100.0 * np.exp(np.cumsum(rets))
        for i, d in enumerate(dates):
            rows.append({"ticker": t, "date": d, "open": close[i],
                         "high": close[i] * 1.001, "low": close[i] * 0.999,
                         "close": close[i], "adj_close": close[i],
                         "volume": 1_000_000.0})
    return pd.DataFrame(rows), dates


def test_analogues_find_repeated_pattern():
    prices, dates = _repeated_pattern_panel()
    # Event frame: single event row for T00 at the end of block 2.
    event_frame = pd.DataFrame([{
        "ticker": "T00", "t0": dates[299], "t1": dates[304],
        "target": 0.02, "rv_1d": 0.0004,
    }])
    # Full frame carries labels for every day (constant except block ends).
    full_rows = []
    for i, d in enumerate(dates):
        full_rows.append({"ticker": "T00", "t0": d,
                          "target": 0.05 if i in (229, 299) else 0.01})
    full_frame = pd.DataFrame(full_rows)
    out = feat_mod.add_attribution_features(event_frame, full_frame, prices)
    # The top analogue of day 299 must be day 229 (identical path,
    # cosine = 1.0). Its label is 0.05 against a 0.01 background, so
    # the top-10 mean is (0.05 + 9 * 0.01) / 10 = 0.014. A mean above
    # 0.013 proves the repeated pattern was found (without it the
    # mean would be exactly the 0.01 background).
    assert out.loc[0, "analogue_vol_mean"] == pytest.approx(0.014, rel=1e-9)
    assert np.isfinite(out.loc[0, "analogue_vol_std"])
    assert np.isfinite(out.loc[0, "driver_H"])
    assert 0.5 <= out.loc[0, "driver_H"] <= 1.0


def test_too_few_analogues_yield_nan():
    # 25-day panel, event on the last day: candidates q in [0, 19],
    # but none has a valid 20-day path (r[0] is NaN poisons the early
    # windows), so zero valid analogues (< 3) -> NaN features.
    prices = _panel(n_days=25, tickers=["T00", "SPY"], seed=9)
    event_frame = pd.DataFrame([{
        "ticker": "T00",
        "t0": pd.DatetimeIndex(sorted(prices["date"].unique()))[24],
        "t1": pd.DatetimeIndex(sorted(prices["date"].unique()))[24],
        "target": 0.02, "rv_1d": 0.0004,
    }])
    full_frame = pd.DataFrame([{
        "ticker": "T00", "t0": d, "target": 0.01,
    } for d in sorted(prices["date"].unique())])
    out = feat_mod.add_attribution_features(event_frame, full_frame, prices)
    assert np.isnan(out.loc[0, "analogue_vol_mean"])
    assert np.isnan(out.loc[0, "analogue_vol_std"])


def _event_frame_for(prices, tickers=("T00",)):
    events = feat_mod.detect_events(prices)
    events = events[events["ticker"].isin(tickers)].reset_index(drop=True)
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    full_rows = [{"ticker": t, "t0": d, "target": 0.012}
                 for t in tickers for d in cal]
    full_frame = pd.DataFrame(full_rows)
    return events, full_frame


def test_point_in_time_truncation():
    """Features for t0 <= cut are identical on the full vs truncated panel."""
    prices = _panel(n_days=300, spike=("T00", 250, 0.15))
    events, full = _event_frame_for(prices)
    full_out = feat_mod.add_attribution_features(events, full, prices)
    cut = pd.Timestamp("2023-01-02")
    trunc_prices = prices[prices["date"] <= cut].reset_index(drop=True)
    trunc_events = events[pd.to_datetime(events["t0"]) <= cut]
    trunc_full = full[pd.to_datetime(full["t0"]) <= cut]
    trunc_out = feat_mod.add_attribution_features(
        trunc_events, trunc_full, trunc_prices)
    merged = full_out.merge(
        trunc_out, on=["ticker", "t0"], suffixes=("_full", "_tr"))
    both = merged[pd.to_datetime(merged["t0"]) <= cut - pd.Timedelta(days=6)]
    assert len(both) > 5
    for c in feat_mod.ATTR_COLUMNS:
        a = both[f"{c}_full"].to_numpy()
        b = both[f"{c}_tr"].to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        assert both_nan.sum() + np.isclose(
            a[~both_nan], b[~both_nan], rtol=1e-9, atol=1e-12).sum() == len(a), c


def test_no_future_cross_sectional_info():
    """Scrambling closes AFTER t0 cannot change that t0's features."""
    prices = _panel(n_days=300, spike=("T00", 250, 0.15))
    events, full = _event_frame_for(prices)
    before = feat_mod.add_attribution_features(events, full, prices)
    cut = pd.Timestamp("2023-06-01")
    rigged = prices.copy()
    future = rigged["date"] > cut
    rigged.loc[future, "adj_close"] = rigged.loc[future, "adj_close"] * 1e6 + 7.0
    events_r, full_r = _event_frame_for(rigged)
    # Recompute events on rigged prices but keep the same event rows by
    # intersecting on (ticker, t0) so the comparison is apples-to-apples.
    common = pd.merge(events[["ticker", "t0"]], events_r[["ticker", "t0"]],
                      on=["ticker", "t0"])
    before_c = before.merge(common, on=["ticker", "t0"])
    after = feat_mod.add_attribution_features(events_r, full_r, rigged)
    after_c = after.merge(common, on=["ticker", "t0"])
    early = before_c["t0"].apply(lambda d: pd.Timestamp(d) <= cut - pd.Timedelta(days=6))
    assert early.sum() > 3
    merged = before_c[early].merge(
        after_c, on=["ticker", "t0"], suffixes=("_b", "_a"))
    for c in feat_mod.ATTR_COLUMNS:
        a = merged[f"{c}_b"].to_numpy()
        b = merged[f"{c}_a"].to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        assert both_nan.sum() + np.isclose(
            a[~both_nan], b[~both_nan], rtol=1e-9, atol=1e-12).sum() == len(a), c


def test_non_trading_day_t0_fails_loud():
    prices = _panel(n_days=300)
    events, full = _event_frame_for(prices)
    assert len(events) > 0
    events = events.copy()
    events.loc[events.index[0], "t0"] = pd.Timestamp("2022-01-08")  # Saturday
    with pytest.raises(ValueError, match="not a price-panel trading day"):
        feat_mod.add_attribution_features(events, full, prices)
