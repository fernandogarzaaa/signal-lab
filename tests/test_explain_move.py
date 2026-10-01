"""Tests for the "Why did it move?" query path. Pure over synthetic frames;
no DB, no network. One integration test uses a temp duckdb + CSV.
"""

import numpy as np
import pandas as pd
import pytest

import signal_lab.rag as ragmod
from signal_lab.explain_move import explain_move, load_news_prices


@pytest.fixture(autouse=True)
def _fake_rag(monkeypatch):
    monkeypatch.setattr(
        ragmod, "explain",
        lambda event: {"text": "10-K context", "citations": [
            {"chunk_id": "AAA#0001", "quote": "revenue grew"}]})


def _prices(ticker="AAA", n=80, start="2026-01-05", closes=None):
    dates = pd.bdate_range(start, periods=n)
    if closes is None:
        closes = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame({
        "ticker": ticker, "date": dates,
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": 1_000_000,
    })


def _news(urls=("u1", "u2", "u3"), pub="2026-04-24 10:00:00"):
    return pd.DataFrame({
        "url": list(urls),
        "title": [f"Headline {u}" for u in urls],
        "body_snippet": ["x" * 300 for _ in urls],
        "published_at": [pub] * len(urls),
        "source_domain": ["example.com"] * len(urls),
    })


def _scored(probas=(0.9, 0.1, 0.55), ticker="AAA", pub="2026-04-24 10:00:00"):
    urls = [f"u{i + 1}" for i in range(len(probas))]
    return pd.DataFrame({
        "url": urls, "text": ["t"] * len(urls), "ticker": [ticker] * len(urls),
        "published_at": [pub] * len(urls), "proba": list(probas),
    })


def test_return_math_on_known_prices():
    px = _prices(n=80)  # closes 100..179, last bdate ~2026-04-24
    event = px["date"].iloc[-1].date().isoformat()
    r = explain_move("AAA", event, _news(), px, _scored())
    assert r["price"]["close"] == pytest.approx(179.0)
    assert r["price"]["prev_close"] == pytest.approx(178.0)
    assert r["price"]["ret_1d"] == pytest.approx(179.0 / 178.0 - 1.0)
    assert r["price"]["ret_3d"] == pytest.approx(179.0 / 176.0 - 1.0)


def test_abnormal_z_math():
    n = 70
    rets = [0.0 if i % 2 else 0.02 for i in range(1, n)]  # 69 alternating returns
    rets[-1] = 0.05  # event-day return
    closes = [100.0]
    for x in rets:
        closes.append(closes[-1] * (1 + x))
    px = _prices(n=n, closes=np.array(closes))
    event = px["date"].iloc[-1].date().isoformat()
    r = explain_move("AAA", event, _news(), px, _scored())
    trail = pd.Series(closes).pct_change().dropna().iloc[-60:-1]
    expected = (0.05 - trail.mean()) / trail.std()
    assert r["price"]["abnormal_z"] == pytest.approx(expected)
    assert r["price"]["is_outlier"] == (abs(expected) > 2.0)


def test_flat_history_gives_no_z_with_note():
    px = _prices(n=80, closes=np.full(80, 100.0))
    event = px["date"].iloc[-1].date().isoformat()
    r = explain_move("AAA", event, _news(), px, _scored())
    assert r["price"]["abnormal_z"] is None
    assert r["price"]["is_outlier"] is False
    assert any("z-score unavailable" in x for x in r["notes"])


def test_driver_ranking_by_signal_strength():
    px = _prices(n=80)
    event = px["date"].iloc[-1].date().isoformat()  # 2026-04-24, news pub same day
    r = explain_move("AAA", event, _news(), px, _scored(), k=5)
    got = [(d["url"], d["direction"], d["proba"]) for d in r["drivers"]]
    # signal: u1=max(.9,.1)=.9, u2=max(.1,.9)=.9, u3=.55; tie broken by proba desc
    assert got == [("u1", "positive", 0.9), ("u2", "negative", 0.1), ("u3", "positive", 0.55)]
    assert r["drivers"][0]["headline"] == "Headline u1"
    assert r["drivers"][0]["source_domain"] == "example.com"
    assert len(r["drivers"][0]["quote"]) == 220  # truncated body_snippet


def test_k_limits_drivers():
    px = _prices(n=80)
    event = px["date"].iloc[-1].date().isoformat()
    r = explain_move("AAA", event, _news(), px, _scored(), k=2)
    assert [d["url"] for d in r["drivers"]] == ["u1", "u2"]


def test_unknown_ticker_raises():
    px = _prices()
    with pytest.raises(ValueError, match="unknown ticker"):
        explain_move("ZZZ", "2026-04-24", _news(), px, _scored())


def test_unparseable_date_raises():
    px = _prices()
    with pytest.raises(ValueError, match="unparseable date"):
        explain_move("AAA", "not-a-date", _news(), px, _scored())


def test_date_outside_coverage_raises():
    px = _prices(n=80, start="2026-01-05")
    with pytest.raises(ValueError, match="outside price coverage"):
        explain_move("AAA", "2020-01-01", _news(), px, _scored())
    with pytest.raises(ValueError, match="outside price coverage"):
        explain_move("AAA", "2030-01-01", _news(), px, _scored())


def test_no_news_window_is_valid_answer_with_note():
    px = _prices(n=80)
    event = px["date"].iloc[-1].date().isoformat()
    sc = _scored(pub="2026-01-02 10:00:00")  # far outside the window
    r = explain_move("AAA", event, _news(pub="2026-01-02 10:00:00"), px, sc)
    assert r["drivers"] == []
    assert any("no news in window" in x for x in r["notes"])
    assert r["timeline"]["markers"] == []


def test_weekend_date_uses_prior_trading_day_with_note():
    px = _prices(n=80, start="2026-01-05")  # bdays; 2026-04-25 is a Saturday
    r = explain_move("AAA", "2026-04-25", _news(), px, _scored())
    assert r["requested_date"] == "2026-04-25"
    assert r["date"] == "2026-04-24"
    assert any("not a trading day" in x for x in r["notes"])


def test_timeline_window_bounds():
    px = _prices(n=80)
    event = px["date"].iloc[40]  # mid-sample so the full [-10, +5] window exists
    pub = event.date().isoformat() + " 10:00:00"
    r = explain_move("AAA", event.date().isoformat(), _news(pub=pub), px, _scored(pub=pub))
    tl = r["timeline"]["prices"]
    assert len(tl) == 16  # 10 before + event day + 5 after
    assert tl[0]["date"] == (event - pd.offsets.BDay(10)).date().isoformat()
    assert tl[-1]["date"] == (event + pd.offsets.BDay(5)).date().isoformat()
    assert tl[10]["date"] == event.date().isoformat()
    assert len(tl[10]) == 2 and set(tl[10]) == {"date", "close"}
    assert r["timeline"]["markers"][0]["headline"] == "Headline u1"


def test_context_note_when_rag_index_missing(monkeypatch):
    def boom(event):
        raise FileNotFoundError("no index")
    monkeypatch.setattr(ragmod, "explain", boom)
    px = _prices(n=80)
    event = px["date"].iloc[-1].date().isoformat()
    r = explain_move("AAA", event, _news(), px, _scored())
    assert r["context"] == {"note": "10-K index not built"}
    assert any("10-K context unavailable" in x for x in r["notes"])


def test_loader_integration_against_temp_stores(tmp_path):
    import duckdb
    db = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE news_raw(url VARCHAR PRIMARY KEY, title VARCHAR, "
                "body_snippet VARCHAR, published_at TIMESTAMP, "
                "source_domain VARCHAR, query VARCHAR, fetched_at TIMESTAMP)")
    con.execute("INSERT INTO news_raw VALUES "
                "('http://x/1', 'Headline One', 'body text here', "
                "'2026-09-15 10:00:00', 'x.com', 'q', '2026-09-15 11:00:00')")
    con.execute("CREATE TABLE prices_daily(ticker VARCHAR, date DATE, open DOUBLE, "
                "high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE)")
    dates = pd.bdate_range("2026-08-01", periods=40)
    closes = 50.0 + np.arange(40, dtype=float)
    for d, c in zip(dates, closes):
        con.execute("INSERT INTO prices_daily VALUES "
                    f"('ZZZ', '{d.date()}', {c}, {c}, {c}, {c}, 1000)")
    con.close()
    csv = tmp_path / "scored.csv"
    pd.DataFrame({"url": ["http://x/1"], "text": ["t"], "ticker": ["ZZZ"],
                  "published_at": ["2026-09-15 10:00:00"], "proba": [0.82]}
                 ).to_csv(csv, index=False)

    news_df, prices_df, scored_df = load_news_prices(db, csv)
    assert len(news_df) == 1 and len(prices_df) == 40 and len(scored_df) == 1
    # 2026-09-15 is a Tuesday and within the bdate range
    r = explain_move("ZZZ", "2026-09-15", news_df, prices_df, scored_df, k=5)
    assert r["ticker"] == "ZZZ"
    assert r["drivers"][0]["headline"] == "Headline One"
    assert r["drivers"][0]["direction"] == "positive"
    assert r["price"]["close"] == pytest.approx(float(closes[dates.get_loc("2026-09-15")]))
