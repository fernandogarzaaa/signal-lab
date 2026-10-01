"""Tests for historical analogues. Pure over synthetic frames; no DB, no network."""

import numpy as np
import pandas as pd
import pytest

from signal_lab.analogues import (
    build_event_index,
    find_analogues,
    load_news_prices,
)


def _prices(ticker="AAA", n=80, start="2026-01-05", closes=None):
    dates = pd.bdate_range(start, periods=n)
    if closes is None:
        # mild alternating wiggle so trailing return std is never zero
        closes = 100.0 + 0.1 * ((np.arange(n) % 2) * 2 - 1)
    return pd.DataFrame({
        "ticker": ticker, "date": dates,
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": 1_000_000,
    })


def _frames(events, ticker="AAA", n=80, start="2026-01-05", base=None):
    """events: list of (day_index, [probas], close). Returns (news, prices, scored).

    `close` overrides that day's close; other days use the mild wiggle.
    """
    px = _prices(ticker=ticker, n=n, start=start)
    if base is not None:
        px["close"] = base(px["close"].to_numpy())
    news_rows, scored_rows = [], []
    for i, (day_idx, probas, close) in enumerate(events):
        if close is not None:
            px.loc[day_idx, "close"] = close
        day = px["date"].iloc[day_idx]
        pub = (day + pd.Timedelta(hours=10)).isoformat()
        for j, p in enumerate(probas):
            url = f"u{i}_{j}"
            news_rows.append({
                "url": url, "title": f"Headline {url}",
                "body_snippet": "x" * 50, "published_at": pub,
                "source_domain": "example.com"})
            scored_rows.append({
                "url": url, "text": "t", "ticker": ticker,
                "published_at": pub, "proba": p})
    news = pd.DataFrame(news_rows)
    scored = pd.DataFrame(scored_rows)
    return news, px, scored


def _day(px, idx):
    return px["date"].iloc[idx].date().isoformat()


def test_event_index_only_days_with_news():
    news, px, scored = _frames([
        (30, [0.9, 0.8], None),
        (50, [0.1], None),
    ])
    idx = build_event_index("AAA", px, scored, news)
    assert {e["date"] for e in idx} == {_day(px, 30), _day(px, 50)}
    by_date = {e["date"]: e for e in idx}
    assert by_date[_day(px, 30)]["n_articles"] == 2
    assert by_date[_day(px, 50)]["mean_proba"] == pytest.approx(0.1)


def test_query_excluded_from_analogues():
    news, px, scored = _frames([
        (30, [0.2], None),
        (50, [0.3], None),
        (70, [0.25], None),  # query
    ])
    r = find_analogues("AAA", _day(px, 70), news, px, scored, k=5)
    assert all(a["date"] != _day(px, 70) for a in r["analogues"])
    assert len(r["analogues"]) == 2


def test_closest_event_ranked_first():
    news, px, scored = _frames([
        (30, [0.9, 0.95], 110.0),    # opposite: up move, positive news
        (50, [0.08, 0.12], 91.0),    # similar: down move, negative news
        (70, [0.05, 0.10], 90.0),    # query: down move, negative news
    ])
    r = find_analogues("AAA", _day(px, 70), news, px, scored, k=2)
    assert r["analogues"][0]["date"] == _day(px, 50)
    assert r["analogues"][1]["date"] == _day(px, 30)
    assert r["analogues"][0]["distance"] < r["analogues"][1]["distance"]


def test_forward_returns_math():
    closes = np.full(80, 100.0)
    news, px, scored = _frames([
        (30, [0.2], None),
        (60, [0.25], None),  # query
    ])
    # deterministic closes: every close is 100 except engineered wiggle is off
    px["close"] = 100.0
    px.loc[5, "close"] = 101.0  # variance seed: trailing std > 0 for all events
    r = find_analogues("AAA", _day(px, 60), news, px, scored, k=1)
    a = r["analogues"][0]
    assert a["date"] == _day(px, 30)
    assert a["forward"]["1d"] == pytest.approx(0.0)
    assert a["forward"]["5d"] == pytest.approx(0.0)
    assert a["forward"]["10d"] == pytest.approx(0.0)


def test_forward_returns_use_future_closes():
    news, px, scored = _frames([(30, [0.2], None), (60, [0.25], None)])
    px["close"] = 100.0
    px.loc[5, "close"] = 101.0  # variance seed: trailing std > 0 for all events
    px.loc[31, "close"] = 105.0   # +5% the day after the analogue
    px.loc[35, "close"] = 110.0
    px.loc[40, "close"] = 90.0
    r = find_analogues("AAA", _day(px, 60), news, px, scored, k=1)
    fwd = r["analogues"][0]["forward"]
    assert fwd["1d"] == pytest.approx(0.05)
    assert fwd["5d"] == pytest.approx(0.10)
    assert fwd["10d"] == pytest.approx(-0.10)


def test_summary_reports_median_mean_frac_positive():
    news, px, scored = _frames([
        (20, [0.2], None),
        (30, [0.2], None),
        (40, [0.2], None),
        (60, [0.25], None),  # query
    ])
    px["close"] = 100.0
    px.loc[5, "close"] = 101.0  # variance seed: trailing std > 0 for all events
    px.loc[21, "close"] = 110.0  # analogue 20: +10% next day
    px.loc[31, "close"] = 90.0   # analogue 30: -10% next day
    px.loc[41, "close"] = 105.0  # analogue 40: +5% next day
    r = find_analogues("AAA", _day(px, 60), news, px, scored, k=3)
    s = r["summary"]["horizons"]["1d"]
    assert s["n"] == 3
    assert s["median"] == pytest.approx(0.05)
    assert s["mean"] == pytest.approx((0.10 - 0.10 + 0.05) / 3)
    assert s["frac_positive"] == pytest.approx(2 / 3)


def test_headline_joined_from_news():
    news, px, scored = _frames([
        (30, [0.05, 0.9], None),  # top-signal article is u0_0 (proba 0.05)
        (60, [0.25], None),
    ])
    r = find_analogues("AAA", _day(px, 60), news, px, scored, k=1)
    assert r["analogues"][0]["headline"] == "Headline u0_0"


def test_unknown_ticker_raises():
    news, px, scored = _frames([(30, [0.2], None)])
    with pytest.raises(ValueError, match="unknown ticker"):
        find_analogues("ZZZ", _day(px, 30), news, px, scored)


def test_bad_date_raises():
    news, px, scored = _frames([(30, [0.2], None)])
    with pytest.raises(ValueError, match="unparseable date"):
        find_analogues("AAA", "not-a-date", news, px, scored)


def test_date_outside_coverage_raises():
    news, px, scored = _frames([(30, [0.2], None)])
    with pytest.raises(ValueError, match="outside price coverage"):
        find_analogues("AAA", "2030-01-01", news, px, scored)


def test_query_without_news_raises_loud():
    news, px, scored = _frames([(30, [0.2], None)])
    with pytest.raises(ValueError, match="no scored news in window"):
        find_analogues("AAA", _day(px, 60), news, px, scored)


def test_k_larger_than_pool_returns_all_with_note():
    news, px, scored = _frames([
        (30, [0.2], None),
        (60, [0.25], None),
    ])
    r = find_analogues("AAA", _day(px, 60), news, px, scored, k=50)
    assert len(r["analogues"]) == 1
    assert any("only 1 past events" in n for n in r["notes"])


def test_non_trading_day_snaps_with_note():
    news, px, scored = _frames([
        (30, [0.2], None),
        (34, [0.25], None),  # a Friday
    ])
    assert px["date"].iloc[34].weekday() == 4
    saturday = (px["date"].iloc[34] + pd.Timedelta(days=1)).date().isoformat()
    r = find_analogues("AAA", saturday, news, px, scored, k=5)
    assert r["date"] == _day(px, 34)
    assert any("not a trading day" in n for n in r["notes"])


def test_truncated_forward_returns_noted():
    news, px, scored = _frames([
        (70, [0.25], None),  # query
        (76, [0.2], None),   # analogue: +10d runs past the end
    ])
    r = find_analogues("AAA", _day(px, 70), news, px, scored, k=5)
    a = r["analogues"][0]
    assert "10d" not in a["forward"]
    assert "1d" in a["forward"]
    assert any("truncated" in n for n in r["notes"])


def test_features_use_no_lookahead():
    # A huge spike AFTER the query must not change the query's abnormal_z.
    news, px, scored = _frames([
        (30, [0.2], None),
        (60, [0.25], None),
    ])
    r1 = find_analogues("AAA", _day(px, 60), news, px, scored, k=1)
    z_before = r1["query"]["abnormal_z"]
    px.loc[65, "close"] = 500.0  # future spike, after the query day
    r2 = find_analogues("AAA", _day(px, 60), news, px, scored, k=1)
    assert r2["query"]["abnormal_z"] == pytest.approx(z_before)


def test_load_news_prices_missing_db_raises(tmp_path):
    with pytest.raises(ValueError, match="database not found"):
        load_news_prices(db_path=tmp_path / "nope.duckdb",
                         scored_path=tmp_path / "nope.csv")


def test_same_burst_analogue_gets_note():
    news, px, scored = _frames([
        (30, [0.2], None),
        (61, [0.25], None),  # adjacent news day after the query: same burst
        (60, [0.22], None),  # query
    ])
    r = find_analogues("AAA", _day(px, 60), news, px, scored, k=5)
    assert any("same news burst" in n for n in r["notes"])
