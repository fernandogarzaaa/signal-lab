"""Tests for the gdelt-client migration (item 5).

The hand-rolled GDELT DOC requests loop is replaced by gdelt-client's
Filters/article_search. These tests cover, without hitting the network:

- the known query template maps to an equivalent Filters object
  (same OR'd phrases, same language, same day bounds, same maxrecords);
- a query that does not match the template fails loudly instead of
  silently changing search semantics;
- the client's article DataFrame maps onto our meta-dict contract;
- empty results and URL-less rows are handled;
- a RateLimitError (429s past retries) propagates so widen.py logs it as
  an explicit fetch_log error gap, never a silent skip;
- retry/backoff configuration is passed through to the client.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from gdelt_client import Filters
from gdelt_client.errors import RateLimitError

from signal_lab.ingest import gdelt_client as gc
from signal_lab.ingest import widen


QUERY = ('("Apple stock" OR "Apple shares" OR "Apple earnings" '
         'OR "AAPL stock" OR "AAPL shares") sourcelang:english')


def test_query_to_filters_matches_template():
    f = gc.query_to_filters(QUERY, "2026-01-01", "2026-01-31", max_records=100)
    qs = f.query_string
    for phrase in ("Apple stock", "Apple shares", "Apple earnings",
                   "AAPL stock", "AAPL shares"):
        assert f'"{phrase}"' in qs
    assert "OR" in qs
    assert "sourcelang:english" in qs
    assert "&maxrecords=100" in qs
    # Same day bounds as the old hand-rolled params: start-of-day..end-of-day.
    assert "&startdatetime=20260101000000" in qs
    assert "&enddatetime=20260131235959" in qs


def test_query_to_filters_rejects_unknown_template():
    with pytest.raises(ValueError, match="does not match the known template"):
        gc.query_to_filters("Apple stock", "2026-01-01", "2026-01-31")


def test_filters_roundtrip_equivalence_with_widen_query():
    # The query widen.build_query produces must map cleanly.
    q = widen.build_query("Apple Inc", "AAPL")
    f = gc.query_to_filters(q, "2023-07-01", "2023-09-30")
    assert isinstance(f, Filters)
    assert '"Apple Inc stock"' in f.query_string


class _FakeClient:
    """Stand-in for GdeltClient capturing constructor kwargs."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        _FakeClient.last_kwargs = kwargs
        self._df = kwargs.pop("_df", None)
        self._exc = kwargs.pop("_exc", None)

    def article_search(self, filters):
        self.filters = filters
        if self._exc is not None:
            raise self._exc
        return self._df


def _article_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "url": "https://example.com/a1",
        "url_mobile": "",
        "title": "Apple beats earnings",
        "seendate": "20260115T143000Z",
        "socialimage": "",
        "domain": "example.com",
        "language": "english",
        "sourcecountry": "US",
    }])


def test_fetch_maps_client_rows_to_meta_contract():
    fake = _FakeClient(_df=_article_df())
    metas = gc.fetch_news_via_client(
        QUERY, "2026-01-01", "2026-01-31", max_records=100,
        fetch_bodies=False, _client_factory=lambda **kw: fake)
    assert len(metas) == 1
    m = metas[0]
    assert m["url"] == "https://example.com/a1"
    assert m["title"] == "Apple beats earnings"
    assert m["published_at"] == datetime(2026, 1, 15, 14, 30,
                                        tzinfo=timezone.utc)
    assert m["source_domain"] == "example.com"
    assert m["query"] == QUERY
    assert m["body_snippet"] == ""
    assert m["fetched_at"] is not None


def test_fetch_empty_results_returns_empty_list():
    fake = _FakeClient(_df=pd.DataFrame())
    metas = gc.fetch_news_via_client(
        QUERY, "2026-01-01", "2026-01-31",
        _client_factory=lambda **kw: fake)
    assert metas == []


def test_fetch_skips_rows_without_url():
    df = pd.DataFrame([
        {"url": "", "title": "no url", "seendate": "20260115T143000Z",
         "domain": "x.com"},
        {"url": "https://example.com/ok", "title": "ok",
         "seendate": "20260115T143000Z", "domain": "example.com"},
    ])
    fake = _FakeClient(_df=df)
    metas = gc.fetch_news_via_client(
        QUERY, "2026-01-01", "2026-01-31", fetch_bodies=False,
        _client_factory=lambda **kw: fake)
    assert [m["url"] for m in metas] == ["https://example.com/ok"]


def test_fetch_bodies_uses_snippet_fn():
    fake = _FakeClient(_df=_article_df())
    metas = gc.fetch_news_via_client(
        QUERY, "2026-01-01", "2026-01-31",
        _fetch_snippet=lambda url: f"snippet for {url}",
        _client_factory=lambda **kw: fake)
    assert metas[0]["body_snippet"] == "snippet for https://example.com/a1"


def test_rate_limit_error_propagates_for_gap_logging():
    # 429s past retries must surface so callers log an explicit fetch gap.
    exc = RateLimitError.__new__(RateLimitError)
    Exception.__init__(exc, "HTTP 429: Too Many Requests")
    fake = _FakeClient(_exc=exc)
    with pytest.raises(RateLimitError):
        gc.fetch_news_via_client(
            QUERY, "2026-01-01", "2026-01-31",
            _client_factory=lambda **kw: fake)


def test_retry_backoff_config_passed_to_client():
    fake = _FakeClient(_df=pd.DataFrame())
    gc.fetch_news_via_client(
        QUERY, "2026-01-01", "2026-01-31", retries=3,
        _client_factory=lambda **kw: _FakeClient(**kw))
    assert _FakeClient.last_kwargs["max_retries"] == 3
    # Politeness posture carried over from the old hand-rolled loop.
    assert _FakeClient.last_kwargs["retry_backoff_base"] == gc.BACKOFF_BASE_S
    assert _FakeClient.last_kwargs["retry_max_wait"] == gc.BACKOFF_MAX_S


def test_rate_limit_surfaces_as_explicit_fetch_log_error(tmp_path, monkeypatch):
    """End-to-end bookkeeping: a throttled quarter is logged status='error'
    with the message, never silently skipped."""
    db = tmp_path / "t.duckdb"
    uni = tmp_path / "universe.csv"
    uni.write_text("ticker,company\nAAA,Acme Corp\n")
    exc = RateLimitError.__new__(RateLimitError)
    Exception.__init__(exc, "HTTP 429: Too Many Requests")

    def boom(query, start, end, max_records=100):
        raise exc

    monkeypatch.setattr(widen, "fetch_prices", lambda *a, **k: pd.DataFrame())
    summary = widen.run_widen(
        db_path=db, universe_path=uni, sleep_s=0, tickers=["AAA"],
        fetch_fn=boom, log=lambda *a, **k: None)
    assert summary["quarters_error"] == summary["pending_before"] > 0
    assert summary["quarters_ok"] == 0
    import duckdb
    con = duckdb.connect(str(db), read_only=True)
    rows = con.execute(
        "SELECT status, error FROM fetch_log").fetchall()
    con.close()
    assert rows and all(s == "error" for s, _ in rows)
    assert all("429" in (e or "") for _, e in rows)
