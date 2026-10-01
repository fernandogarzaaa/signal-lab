"""Tests for signal_lab.ingest.widen (gate-2 data widening)."""

from __future__ import annotations

import csv
from datetime import date, datetime, timezone

import duckdb
import pytest

import signal_lab.ingest as ingest
from signal_lab.ingest import widen


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    db = str(tmp_path / "test.duckdb")
    monkeypatch.setattr(ingest, "DB_PATH", db)
    return db


def _uni(tmp_path) -> str:
    p = tmp_path / "uni.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "company"])
        w.writerow(["AAA", "Aaa Corp"])
        w.writerow(["BBB", "Bbb Inc"])
    return str(p)


def test_quarter_windows_defaults():
    ws = widen.quarter_windows()
    assert len(ws) == 12
    assert ws[0] == (date(2023, 7, 1), date(2023, 9, 30))
    assert ws[-1] == (date(2026, 4, 1), date(2026, 6, 30))
    # contiguous, no gaps or overlaps
    for (s1, e1), (s2, e2) in zip(ws, ws[1:]):
        assert (e1 - s1).days >= 27
        assert s2 == e1 + (date(2023, 1, 2) - date(2023, 1, 1))


def test_quarter_windows_partial():
    ws = widen.quarter_windows(date(2023, 8, 15), date(2024, 2, 10))
    assert ws[0] == (date(2023, 8, 15), date(2023, 9, 30))
    assert ws[-1] == (date(2024, 1, 1), date(2024, 2, 10))
    assert len(ws) == 3


def test_quarter_windows_bad_range():
    with pytest.raises(ValueError):
        widen.quarter_windows(date(2024, 1, 1), date(2023, 1, 1))


def test_build_query():
    q = widen.build_query("Apple", "AAPL")
    assert q == ('("Apple stock" OR "Apple shares" OR "Apple earnings" '
                 'OR "AAPL stock" OR "AAPL shares") sourcelang:english')


def test_load_universe_real():
    uni = widen.load_universe()
    assert len(uni) == 100
    tickers = [t for t, _ in uni]
    assert len(set(tickers)) == 100
    assert ("AAPL", "Apple") in uni


def test_load_universe_validates(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("ticker,company\nAAA,Aaa\nAAA,Aaa dup\n")
    with pytest.raises(ValueError, match="duplicate"):
        widen.load_universe(p)
    with pytest.raises(ValueError, match="not found"):
        widen.load_universe(tmp_path / "nope.csv")


def test_pending_quarters_skips_ok_retries_error(tmp_db, tmp_path):
    uni = [("AAA", "Aaa Corp"), ("BBB", "Bbb Inc")]
    con = duckdb.connect(tmp_db)
    widen.ensure_fetch_log(con)
    assert len(widen.pending_quarters(con, uni)) == 24
    qs = widen.quarter_windows()[0][0]
    widen._log(con, "AAA", qs, date(2023, 9, 30), "q", 10, "ok", None)
    widen._log(con, "BBB", qs, date(2023, 9, 30), "q", 0, "error", "boom")
    pend = widen.pending_quarters(con, uni)
    assert len(pend) == 23
    # the ok quarter is gone; the errored one is back for retry
    assert not any(t == "AAA" and q == qs for t, _, q, _ in pend)
    assert any(t == "BBB" and q == qs for t, _, q, _ in pend)
    con.close()


def _fake_fetch(query, start, end, max_records=100):
    import hashlib
    now = datetime.now(timezone.utc)
    tag = hashlib.md5(query.encode()).hexdigest()[:8]
    return [{
        "url": f"https://example.com/{tag}/{start}/{i}",
        "title": f"Test article {i} {query[:20]}",
        "body_snippet": "snippet",
        "published_at": f"{start}T12:00:00Z",
        "source_domain": "example.com",
        "query": query,
        "fetched_at": now,
    } for i in range(3)]


def test_run_widen_end_to_end_and_idempotent(tmp_db, tmp_path, monkeypatch):
    uni_path = _uni(tmp_path)
    # stub out price fetching (no network in tests)
    monkeypatch.setattr(widen, "fetch_prices",
                        lambda tickers, s, e: __import__("pandas").DataFrame(
                            {"ticker": [], "date": []}))
    s1 = widen.run_widen(tmp_db, uni_path, max_records=10, sleep_s=0,
                         fetch_fn=_fake_fetch, log=lambda *a: None)
    assert s1["quarters_ok"] == 24 and s1["quarters_error"] == 0
    con = duckdb.connect(tmp_db, read_only=True)
    n = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
    assert n == 24 * 3  # 2 tickers x 12 quarters x 3 articles
    con.close()
    # second run: nothing pending
    s2 = widen.run_widen(tmp_db, uni_path, max_records=10, sleep_s=0,
                         fetch_fn=_fake_fetch, log=lambda *a: None)
    assert s2["pending_before"] == 0 and s2["quarters_ok"] == 0


def test_run_widen_logs_errors_and_continues(tmp_db, tmp_path, monkeypatch):
    uni_path = _uni(tmp_path)
    monkeypatch.setattr(widen, "fetch_prices",
                        lambda tickers, s, e: __import__("pandas").DataFrame(
                            {"ticker": [], "date": []}))
    calls = {"n": 0}

    def flaky(query, start, end, max_records=100):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return _fake_fetch(query, start, end, max_records)

    s = widen.run_widen(tmp_db, uni_path, sleep_s=0, fetch_fn=flaky,
                        log=lambda *a: None)
    assert s["quarters_ok"] == 23 and s["quarters_error"] == 1
    con = duckdb.connect(tmp_db, read_only=True)
    errs = con.execute(
        "SELECT error FROM fetch_log WHERE status='error'").fetchall()
    assert len(errs) == 1 and "transient" in errs[0][0]
    con.close()
    # retry recovers the failed quarter
    s2 = widen.run_widen(tmp_db, uni_path, sleep_s=0, fetch_fn=_fake_fetch,
                         log=lambda *a: None)
    assert s2["quarters_ok"] == 1 and s2["quarters_error"] == 0


def test_quality_check_fails_loudly(tmp_db, tmp_path, monkeypatch):
    uni_path = _uni(tmp_path)
    monkeypatch.setattr(widen, "fetch_prices",
                        lambda tickers, s, e: __import__("pandas").DataFrame(
                            {"ticker": [], "date": []}))
    # nothing fetched yet
    with pytest.raises(ValueError, match="database not found"):
        widen.quality_check(tmp_db, uni_path)
    widen.run_widen(tmp_db, uni_path, sleep_s=0, fetch_fn=_fake_fetch,
                    log=lambda *a: None)
    # prices missing -> loud failure
    with pytest.raises(ValueError, match="no price rows"):
        widen.quality_check(tmp_db, uni_path)
    # add price coverage for both tickers through the required range
    import pandas as pd
    px = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "date": [date(2026, 7, 10), date(2026, 7, 10)],
        "open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0],
        "close": [1.0, 2.0], "volume": [100, 200],
    })
    ingest.store_prices(px)
    out = widen.quality_check(tmp_db, uni_path)
    assert out["tickers"] == 2
    assert out["total_articles_fetched"] == 24 * 3
    assert set(out["per_ticker_articles"]) == {"AAA", "BBB"}
