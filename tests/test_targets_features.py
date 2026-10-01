"""Tests for Phase 2: versioned targets and price/volume features.

Every target and indicator is verified against hand-computed values on
synthetic OHLCV panels: the expected numbers below are derived by hand
(the arithmetic is shown in the comments) and written as literals or
fractions, never computed by the implementation under test. The
no-leakage tripwire pins the core guarantee: post-publication price moves
must not change any feature.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signal_lab.models.context_features import (
    PRICE_FEATURE_NAMES,
    _ema,
    build_price_features,
)
from signal_lab.models.labels import LabelConfig, forward_cumulative_abnormal
from signal_lab.validation.point_in_time import FEATURE_POINT_IN_TIME
from signal_lab.validation.targets import (
    TARGET_SET_VERSION,
    TargetConfig,
    build_targets,
    target_column_names,
)

QUIET = {"log": lambda *a, **k: None}
DB = Path(__file__).resolve().parents[1] / "data" / "signal_lab.duckdb"


def _ohlcv(ticker, closes, start="2026-01-05", volume=1000.0, spread=1.0,
           high=None, low=None):
    n = len(closes)
    dates = pd.bdate_range(start, periods=n)
    closes = [float(c) for c in closes]
    vols = volume if isinstance(volume, list) else [volume] * n
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": [d.date() for d in dates],
            "open": closes,
            "high": [float(h) for h in high]
            if high is not None
            else [c + spread for c in closes],
            "low": [float(v) for v in low]
            if low is not None
            else [c - spread for c in closes],
            "close": closes,
            "volume": [float(v) for v in vols],
        }
    )


def _panel(*frames):
    return pd.concat(frames, ignore_index=True)


def _news_row(url, ticker, pub_date, t0=None):
    row = {
        "url": url,
        "ticker": ticker,
        "published_at": pd.Timestamp(pub_date, tz="UTC"),
        "pub_date": pub_date,
        "text": "some market news text here",
    }
    if t0 is not None:
        row["t0"] = t0
    return row


# ---------------------------------------------------------------------------
# Target config
# ---------------------------------------------------------------------------


def test_target_config_validation():
    TargetConfig().validated()
    assert TargetConfig(horizons=[1, 3]).validated().horizons == (1, 3)
    with pytest.raises(ValueError):
        TargetConfig(horizons=()).validated()
    with pytest.raises(ValueError):
        TargetConfig(horizons=(0, 3)).validated()
    with pytest.raises(ValueError):
        TargetConfig(horizons=(3, 3)).validated()
    with pytest.raises(ValueError):
        TargetConfig(horizons=(1.5,)).validated()
    with pytest.raises(ValueError):
        TargetConfig(benchmark="").validated()
    assert target_column_names(TargetConfig().validated()) == [
        "target_version",
        "ret_1d", "mkt_1d", "excess_1d", "direction_1d",
        "ret_3d", "mkt_3d", "excess_3d", "direction_3d",
        "ret_5d", "mkt_5d", "excess_5d", "direction_5d",
        "realized_vol_5d",
    ]


# ---------------------------------------------------------------------------
# Targets: hand-computed values
# ---------------------------------------------------------------------------


def test_targets_hand_computed():
    # AAA closes 100..108 (9 bars d0..d8), SPY flat at 500. Article t0 = d2
    # (close 102). Window (close(d2), close(d2+H)]:
    #   ret_1d = 103/102 - 1 = 1/102
    #   ret_3d = 105/102 - 1 = 1/34
    #   ret_5d = 107/102 - 1 = 5/102
    # mkt_* = 0 (flat benchmark), excess_* = ret_*, direction_* = 1.
    dates = pd.bdate_range("2026-01-05", periods=9)
    prices = _panel(
        _ohlcv("AAA", [100 + i for i in range(9)]),
        _ohlcv("SPY", [500.0] * 9),
    )
    news = pd.DataFrame([{"ticker": "AAA", "t0": dates[2].date()}])
    tgt = build_targets(news, prices, TargetConfig().validated(), **QUIET)
    assert (tgt["target_version"] == TARGET_SET_VERSION).all()
    assert tgt["ret_1d"].iloc[0] == pytest.approx(1 / 102)
    assert tgt["ret_3d"].iloc[0] == pytest.approx(1 / 34)
    assert tgt["ret_5d"].iloc[0] == pytest.approx(5 / 102)
    for h in (1, 3, 5):
        assert tgt[f"mkt_{h}d"].iloc[0] == pytest.approx(0.0)
        assert tgt[f"excess_{h}d"].iloc[0] == pytest.approx(
            tgt[f"ret_{h}d"].iloc[0]
        )
        assert tgt[f"direction_{h}d"].iloc[0] == 1.0
    assert np.isfinite(tgt["realized_vol_5d"].iloc[0])


def test_t0_anchoring_spike_into_t0_excluded():
    # A spike that lands exactly ON t0's close must not enter the window:
    # the window base is close(t0), so the 50 -> 100 move into d0 is
    # excluded and ret_1d = 100/100 - 1 = 0.
    dates = pd.bdate_range("2026-01-05", periods=8)
    prices = _panel(
        _ohlcv("AAA", [50.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]),
        _ohlcv("SPY", [500.0] * 8),
    )
    news = pd.DataFrame([{"ticker": "AAA", "t0": dates[1].date()}])
    tgt = build_targets(news, prices, TargetConfig().validated(), **QUIET)
    assert tgt["ret_1d"].iloc[0] == pytest.approx(0.0)
    assert tgt["excess_1d"].iloc[0] == pytest.approx(0.0)
    assert tgt["direction_1d"].iloc[0] == 0.0  # not > 0


def test_t0_anchoring_after_close_publication():
    # End-to-end through the real t0 path: published Tue 2026-01-06 18:00
    # ET (after the close) -> t0 is Tuesday itself, so the label window is
    # (close Tue, close Wed] and ret_1d = close[2]/close[1] - 1.
    from signal_lab.validation.timing import compute_t0

    dates = pd.bdate_range("2026-01-05", periods=10)
    closes = [100.0 + i for i in range(10)]
    prices = _panel(
        _ohlcv("AAA", closes), _ohlcv("SPY", [500.0] * 10)
    )
    pub_utc = pd.Timestamp("2026-01-06 18:00", tz="America/New_York").tz_convert(
        "UTC"
    )
    news = pd.DataFrame([{"ticker": "AAA", "published_at": pub_utc}])
    cals = {
        t: np.array([d.date() for d in dates]) for t in ("AAA", "SPY")
    }
    t0 = compute_t0(news["published_at"], cals, news["ticker"].to_numpy())
    assert t0.iloc[0] == dates[1].date()
    news["t0"] = t0
    tgt = build_targets(news, prices, TargetConfig().validated(), **QUIET)
    assert tgt["ret_1d"].iloc[0] == pytest.approx(closes[2] / closes[1] - 1)
    assert tgt["ret_3d"].iloc[0] == pytest.approx(closes[4] / closes[1] - 1)


def test_partial_window_nan_row_kept():
    # 9 bars d0..d8, t0 = d7: H=1 is complete, H=3/H=5 are partial ->
    # NaN for those horizons' columns, row kept. realized_vol_5d needs
    # d8..d12 -> NaN. A missing t0 NaNs everything.
    dates = pd.bdate_range("2026-01-05", periods=9)
    prices = _panel(
        _ohlcv("AAA", [100 + i for i in range(9)]),
        _ohlcv("SPY", [500.0] * 9),
    )
    news = pd.DataFrame(
        [
            {"ticker": "AAA", "t0": dates[7].date()},
            {"ticker": "AAA", "t0": None},
        ]
    )
    tgt = build_targets(news, prices, TargetConfig().validated(), **QUIET)
    assert len(tgt) == 2  # rows kept, not dropped
    r = tgt.iloc[0]
    assert r["ret_1d"] == pytest.approx(108 / 107 - 1)
    assert r["direction_1d"] == 1.0
    for h in (3, 5):
        for c in (f"ret_{h}d", f"mkt_{h}d", f"excess_{h}d", f"direction_{h}d"):
            assert np.isnan(r[c]), c
    assert np.isnan(r["realized_vol_5d"])
    r2 = tgt.iloc[1]
    for c in target_column_names(TargetConfig().validated()):
        if c == "target_version":
            assert r2[c] == TARGET_SET_VERSION
        else:
            assert np.isnan(r2[c]), c


def test_realized_vol_5d_hand_computed():
    # Geometric closes give exact log returns [0.01,-0.01,0.01,-0.01,0.01].
    # mean = 0.002; sq dev = 3*(0.008^2) + 2*(0.012^2) = 0.00048;
    # sample var = 0.00048/4 = 0.00012; std = sqrt(0.00012).
    rets = [0.01, -0.01, 0.01, -0.01, 0.01]
    closes = [100.0]
    for r in rets:
        closes.append(closes[-1] * math.exp(r))
    dates = pd.bdate_range("2026-01-05", periods=6)
    prices = _panel(_ohlcv("AAA", closes), _ohlcv("SPY", [500.0] * 6))
    news = pd.DataFrame([{"ticker": "AAA", "t0": dates[0].date()}])
    tgt = build_targets(news, prices, TargetConfig().validated(), **QUIET)
    assert tgt["realized_vol_5d"].iloc[0] == pytest.approx(
        math.sqrt(0.00012), abs=1e-9
    )
    assert tgt["ret_5d"].iloc[0] == pytest.approx(math.exp(0.01) - 1)
    assert tgt["direction_5d"].iloc[0] == 1.0


def test_excess_matches_forward_cumulative_abnormal():
    # Formula identity with models.labels on a synthetic panel where SPY
    # misses one date (exercises the universe-mean market fallback).
    dates = pd.bdate_range("2026-01-05", periods=12)
    spy = _ohlcv("SPY", [500.0] * 12)
    spy = spy[spy["date"] != dates[5].date()].reset_index(drop=True)
    prices = _panel(
        _ohlcv("AAA", [100 + i for i in range(12)]),
        _ohlcv("BBB", [200 - 2 * i for i in range(12)]),
        spy,
    )
    news = pd.DataFrame(
        [
            {"ticker": "AAA", "t0": dates[2].date()},
            {"ticker": "BBB", "t0": dates[5].date()},
            {"ticker": "AAA", "t0": dates[9].date()},  # partial for H>=3
        ]
    )
    tgt = build_targets(news, prices, TargetConfig().validated(), **QUIET)
    for h in (1, 3, 5):
        fwd = forward_cumulative_abnormal(
            prices, LabelConfig(window_days=h, benchmark="SPY")
        )
        m = news.merge(
            fwd, left_on=["ticker", "t0"], right_on=["ticker", "date"], how="left"
        )
        partial = m["fwd_days"].to_numpy() < h
        for ours, theirs in (
            (f"ret_{h}d", "ticker_fwd_ret"),
            (f"mkt_{h}d", "mkt_fwd_ret"),
            (f"excess_{h}d", "abn_ret"),
        ):
            a = tgt[ours].to_numpy()
            b = m[theirs].to_numpy()
            assert np.isnan(a[partial]).all(), (h, ours)
            np.testing.assert_allclose(
                a[~partial], b[~partial], rtol=1e-12, atol=1e-12,
                err_msg=f"H={h} {ours}",
            )
        d = tgt[f"direction_{h}d"].to_numpy()
        assert np.isnan(d[partial]).all()
        np.testing.assert_array_equal(
            d[~partial], (m["ticker_fwd_ret"].to_numpy()[~partial] > 0).astype(float)
        )


@pytest.mark.skipif(not DB.exists(), reason="requires the real DuckDB backfill")
def test_excess_3d_equals_abn_ret_real_data():
    from signal_lab.models.build_and_train import build_dataset

    df, _ = build_dataset(log=lambda *a, **k: None)
    assert len(df) > 0
    assert (df["target_version"] == TARGET_SET_VERSION).all()
    for c in target_column_names(TargetConfig().validated()) + PRICE_FEATURE_NAMES:
        assert c in df.columns, c
    diff = (df["excess_3d"] - df["abn_ret"]).abs()
    assert float(diff.max()) < 1e-12


# ---------------------------------------------------------------------------
# Price features: hand-computed values
# ---------------------------------------------------------------------------


def test_ema_hand_computed():
    # alpha = 2/13. e[0] = 100; e[1] = (2*110 + 11*100)/13 = 1320/13;
    # e[2] = (2*120 + 11*(1320/13))/13 = 17640/169.
    e = _ema(np.array([100.0, 110.0, 120.0]), span=12)
    assert e[0] == pytest.approx(100.0)
    assert e[1] == pytest.approx(1320 / 13)
    assert e[2] == pytest.approx(17640 / 169)


def test_rsi_hand_computed():
    # 16 bars: fourteen +2 deltas then a -14 delta.
    # Seed at bar 14: avg_gain = 2, avg_loss = 0 -> RSI = 100.
    # Bar 15: avg_gain = (2*13 + 0)/14 = 26/14, avg_loss = (0*13 + 14)/14 = 1,
    #   RSI = 100 - 100/(1 + 26/14) = 100 - 35 = 65.
    closes = [100.0 + 2 * i for i in range(15)] + [114.0]
    dates = pd.bdate_range("2026-01-05", periods=17)
    prices = _panel(
        _ohlcv("AAA", closes), _ohlcv("SPY", [500.0] * 16)
    )
    news = pd.DataFrame(
        [
            _news_row("https://x.com/r1", "AAA", dates[15].date()),
            _news_row("https://x.com/r2", "AAA", dates[16].date()),
        ]
    )
    out = build_price_features(news, prices, **QUIET)
    assert out["ctx_asof"].iloc[0] == dates[14].date()
    assert out["ctx_asof"].iloc[1] == dates[15].date()
    assert out["px_rsi_14"].iloc[0] == pytest.approx(100.0)
    assert out["px_rsi_14"].iloc[1] == pytest.approx(65.0)


def test_macd_hist_flat_series_is_zero():
    dates = pd.bdate_range("2026-01-05", periods=41)
    prices = _panel(
        _ohlcv("AAA", [100.0] * 40), _ohlcv("SPY", [500.0] * 40)
    )
    news = pd.DataFrame([_news_row("https://x.com/m", "AAA", dates[40].date())])
    out = build_price_features(news, prices, **QUIET)
    assert out["px_macd_hist"].iloc[0] == pytest.approx(0.0)


def test_macd_hist_hand_computed():
    # 37 bars: 34 flat at 100, then 100/110/120 at bars 34/35/36.
    # Flat prefix -> all EMAs = 100 at bar 34.
    # Bar 35: e12 = 1320/13, e26 = 2720/27, m = 280/351, s = 56/351.
    # Bar 36: e12 = 17640/169, e26 = 74480/729,
    #   m36 = 17640/169 - 74480/729, s36 = 0.2*m36 + 0.8*56/351,
    #   hist = m36 - s36 = 0.8*(m36 - 56/351).
    closes = [100.0] * 34 + [100.0, 110.0, 120.0]
    dates = pd.bdate_range("2026-01-05", periods=38)
    prices = _panel(
        _ohlcv("AAA", closes), _ohlcv("SPY", [500.0] * 37)
    )
    news = pd.DataFrame([_news_row("https://x.com/m", "AAA", dates[37].date())])
    out = build_price_features(news, prices, **QUIET)
    expected = 0.8 * ((17640 / 169 - 74480 / 729) - 56 / 351)
    assert out["px_macd_hist"].iloc[0] == pytest.approx(expected)


def test_atr_hand_computed():
    # Bars 0..14: TR = max(4, |102-100|, |98-100|) = 4 -> ATR = 4.
    # Bar 15: TR = max(20, |110-100|, |90-100|) = 20 ->
    #   ATR = (4*13 + 20)/14 = 72/14; px_atr_14 = (72/14)/100.
    n = 16
    dates = pd.bdate_range("2026-01-05", periods=17)
    prices = _panel(
        _ohlcv(
            "AAA", [100.0] * n,
            high=[102.0] * 15 + [110.0], low=[98.0] * 15 + [90.0],
        ),
        _ohlcv("SPY", [500.0] * n),
    )
    news = pd.DataFrame(
        [
            _news_row("https://x.com/a1", "AAA", dates[15].date()),
            _news_row("https://x.com/a2", "AAA", dates[16].date()),
        ]
    )
    out = build_price_features(news, prices, **QUIET)
    assert out["px_atr_14"].iloc[0] == pytest.approx(4 / 100)
    assert out["px_atr_14"].iloc[1] == pytest.approx((72 / 14) / 100)


def test_ma_dist_and_ret_1d_hand_computed():
    # Closes 1..50, as-of bar 49 (close 50):
    #   px_ret_1d = 50/49 - 1 = 1/49
    #   SMA20 = mean(31..50) = 40.5 -> dist = 9.5/40.5 = 19/81
    #   SMA50 = mean(1..50) = 25.5 -> dist = 24.5/25.5 = 49/51
    dates = pd.bdate_range("2026-01-05", periods=51)
    prices = _panel(
        _ohlcv("AAA", list(range(1, 51))), _ohlcv("SPY", [500.0] * 50)
    )
    news = pd.DataFrame([_news_row("https://x.com/m", "AAA", dates[50].date())])
    out = build_price_features(news, prices, **QUIET)
    assert out["px_ret_1d"].iloc[0] == pytest.approx(1 / 49)
    assert out["px_ma_dist_20"].iloc[0] == pytest.approx(19 / 81)
    assert out["px_ma_dist_50"].iloc[0] == pytest.approx(49 / 51)


def test_vol_z_and_relvol_hand_computed():
    # 20 bars, volumes 19x1000 then 3000 at D:
    #   mean = 1100; sum sq dev = 19*100^2 + 1900^2 = 3,800,000;
    #   var = 3,800,000/19 = 200,000; z = 1900/sqrt(200000).
    #   relvol = 3000/1000 = 3.
    dates = pd.bdate_range("2026-01-05", periods=21)
    prices = _panel(
        _ohlcv("AAA", [100.0] * 20, volume=[1000.0] * 19 + [3000.0]),
        _ohlcv("SPY", [500.0] * 20),
    )
    news = pd.DataFrame([_news_row("https://x.com/v", "AAA", dates[20].date())])
    out = build_price_features(news, prices, **QUIET)
    assert out["px_vol_z_20"].iloc[0] == pytest.approx(1900 / math.sqrt(200000))
    assert out["px_relvol_20"].iloc[0] == pytest.approx(3.0)


def test_vol_z_zero_on_constant_volume():
    dates = pd.bdate_range("2026-01-05", periods=21)
    prices = _panel(
        _ohlcv("AAA", [100.0] * 20), _ohlcv("SPY", [500.0] * 20)
    )
    news = pd.DataFrame([_news_row("https://x.com/v", "AAA", dates[20].date())])
    out = build_price_features(news, prices, **QUIET)
    assert out["px_vol_z_20"].iloc[0] == pytest.approx(0.0)
    assert out["px_relvol_20"].iloc[0] == pytest.approx(1.0)


def test_vs_spy_and_market_features_hand_computed():
    # AAA flat at 100; SPY flat at 100 for bars 0..24, then
    # 102/104/106/108/110 at bars 25..29. As-of bar 29:
    #   px_vs_spy_5d = 0 - (110/100 - 1) = -0.1
    #   px_vs_spy_20d = 0 - (110/100 - 1) = -0.1
    #   mkt_spy_ret_20d = 110/100 - 1 = 0.1
    dates = pd.bdate_range("2026-01-05", periods=31)
    spy_closes = [100.0] * 25 + [102.0, 104.0, 106.0, 108.0, 110.0]
    prices = _panel(
        _ohlcv("AAA", [100.0] * 30), _ohlcv("SPY", spy_closes)
    )
    news = pd.DataFrame([_news_row("https://x.com/s", "AAA", dates[30].date())])
    out = build_price_features(news, prices, **QUIET)
    assert out["px_vs_spy_5d"].iloc[0] == pytest.approx(-0.1)
    assert out["px_vs_spy_20d"].iloc[0] == pytest.approx(-0.1)
    assert out["mkt_spy_ret_20d"].iloc[0] == pytest.approx(0.1)
    assert np.isfinite(out["mkt_spy_vol_20d"].iloc[0])
    assert out["mkt_spy_vol_20d"].iloc[0] > 0


def test_insufficient_history_is_nan_not_filled():
    # 10 bars only: long-warmup features NaN, short ones computable.
    # The NaN must survive even though the panel holds later bars for
    # other articles (no forward-fill across the as-of boundary).
    dates = pd.bdate_range("2026-01-05", periods=60)
    prices = _panel(
        _ohlcv("AAA", [100.0 + i for i in range(60)]),
        _ohlcv("SPY", [500.0] * 60),
    )
    news = pd.DataFrame(
        [
            _news_row("https://x.com/e", "AAA", dates[10].date()),  # as-of bar 9
            _news_row("https://x.com/l", "AAA", dates[59].date()),  # as-of bar 58
        ]
    )
    out = build_price_features(news, prices, **QUIET)
    early, late = out.iloc[0], out.iloc[1]
    for c in (
        "px_rsi_14", "px_macd_hist", "px_ma_dist_20", "px_ma_dist_50",
        "px_realvol_20d", "px_atr_14", "px_vol_z_20", "px_relvol_20",
    ):
        assert np.isnan(early[c]), c
        assert np.isfinite(late[c]), c
    # Short warmups are computable even for the early article.
    assert np.isfinite(early["px_ret_1d"])
    assert np.isfinite(early["px_realvol_5d"])
    assert np.isfinite(early["px_vs_spy_5d"])


def test_benchmark_missing_leaves_spy_features_nan():
    dates = pd.bdate_range("2026-01-05", periods=41)
    prices = _ohlcv("AAA", [100.0 + i for i in range(40)])
    news = pd.DataFrame([_news_row("https://x.com/b", "AAA", dates[40].date())])
    out = build_price_features(news, prices, benchmark="SPY", **QUIET)
    for c in (
        "px_vs_spy_5d", "px_vs_spy_20d", "mkt_spy_ret_20d", "mkt_spy_vol_20d",
    ):
        assert np.isnan(out[c].iloc[0]), c
    assert np.isfinite(out["px_ret_1d"].iloc[0])


def test_no_leakage_post_publication_spike():
    # Article published on the calendar date of trading day 60 (as-of day
    # 59). A 3x price AND volume spike on days 70+ must not move any of
    # the 14 price features; a spike on day 50 (before the as-of day)
    # must move them, proving the test is not vacuous.
    closes = [100.0 + i for i in range(80)]
    prices = _panel(
        _ohlcv("BBB", closes), _ohlcv("SPY", [500.0] * 80)
    )
    pub = pd.bdate_range("2026-01-05", periods=80)[60].date()
    news = pd.DataFrame([_news_row("https://x.com/a", "BBB", pub)])
    base = build_price_features(news, prices, **QUIET)

    # Sanity: the as-of day really is trading day 59.
    assert base["px_ret_1d"].iloc[0] == pytest.approx(159.0 / 158.0 - 1.0)

    spiked = prices.copy()
    late = spiked["date"] >= spiked["date"].iloc[70]
    for col in ("close", "high", "low", "volume"):
        spiked.loc[late, col] *= 3.0
    after = build_price_features(news, spiked, **QUIET)
    pd.testing.assert_frame_equal(
        base[PRICE_FEATURE_NAMES], after[PRICE_FEATURE_NAMES], check_exact=True
    )

    early = prices.copy()
    pre = early["date"] >= early["date"].iloc[50]
    for col in ("close", "high", "low", "volume"):
        early.loc[pre, col] *= 3.0
    moved = build_price_features(news, early, **QUIET)
    assert not moved[PRICE_FEATURE_NAMES].equals(base[PRICE_FEATURE_NAMES])


def test_registry_covers_price_features():
    for name in PRICE_FEATURE_NAMES:
        entry = FEATURE_POINT_IN_TIME[name]  # KeyError if unregistered
        assert {"observation", "publication", "availability", "measures"} <= set(
            entry
        ), name
