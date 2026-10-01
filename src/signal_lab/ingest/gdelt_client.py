"""GDELT DOC API fetching via the maintained ``gdelt-client`` package.

This module replaces the hand-rolled ``requests`` loop that used to live in
``signal_lab.ingest.fetch_news``.

What this migration is and is not:
- IS a code-health migration: a maintained query builder (``Filters``),
  typed errors (``RateLimitError`` for 429s), and tenacity-backed retries
  on 429/5xx/connection errors instead of our ad-hoc backoff loop.
- IS NOT a throttle fix: the DOC API hard-throttled this IP during gate-2
  (HTTP 429s), and no client change alters provider rate limits. We keep
  our politeness posture: long backoff waits (15s base, 120s cap) and the
  30s sleeps between queries in the callers (``run.py``, ``widen.py``).
- Checkpointing and explicit gap logging stay ours: ``widen.py``'s
  ``fetch_log`` records every (ticker, quarter) as ok/error, and failed
  quarters are retried, never silently skipped.

Query mapping: every caller builds queries from the same template
``("{company} stock" OR ... ) sourcelang:english``. ``query_to_filters``
parses that template strictly into ``Filters(keyword=[...phrases],
language=...)``; any query that does not match the known template raises
``ValueError`` instead of silently changing search semantics.

Two documented behavioral differences vs the old hand-rolled code:
- Sort order: the old code passed ``sort=datedesc``; gdelt-client does not
  expose a sort parameter, so results arrive in the API's default order.
  Nothing downstream depends on ordering (``store_news`` upserts by URL).
- Date bounds are preserved exactly: start-of-day .. end-of-day
  (``YYYYMMDD000000`` .. ``YYYYMMD235959``), same as the old params.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from gdelt_client import Filters, GdeltClient
from gdelt_client.errors import RateLimitError

# Strict template: ("phrase1" OR "phrase2" ...) sourcelang:xx
_TEMPLATE_RE = re.compile(r'^\("([^"]+)"((?: OR "[^"]+")*)\) sourcelang:([A-Za-z]+)$')
_OR_PHRASE_RE = re.compile(r'OR "([^"]+)"')

# Backoff posture carried over from the old hand-rolled loop (15 * 2**n).
BACKOFF_BASE_S = 15
BACKOFF_MAX_S = 120


def query_to_filters(query: str, start: str, end: str,
                     max_records: int = 250) -> Filters:
    """Translate our known query template into a gdelt-client Filters.

    Raises ValueError if the query does not match the template: a silent
    semantic change to the search would corrupt the news corpus.
    """
    m = _TEMPLATE_RE.match(query.strip())
    if not m:
        raise ValueError(
            "[gdelt] query does not match the known template "
            f"(\"(...)\" OR ...) sourcelang:xx; refusing to guess: {query[:80]!r}")
    phrases = [m.group(1)] + _OR_PHRASE_RE.findall(m.group(2))
    language = m.group(3)
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = (datetime.strptime(end, "%Y-%m-%d")
              + timedelta(days=1) - timedelta(seconds=1)).replace(tzinfo=timezone.utc)
    return Filters(
        keyword=phrases,
        language=language,
        start_date=start_dt,
        end_date=end_dt,
        num_records=max_records,
    )


def _row_to_meta(row: dict, query: str, fetched_at: datetime,
                 snippet: str) -> dict:
    return {
        "url": row.get("url") or "",
        "title": row.get("title") or "",
        "published_at": _parse_gdelt_dt(str(row.get("seendate") or "")),
        "source_domain": row.get("domain") or "",
        "query": query,
        "fetched_at": fetched_at,
        "body_snippet": snippet,
    }


def _parse_gdelt_dt(s: str) -> datetime | None:
    # GDELT seendate looks like "20260930T120000Z"
    try:
        return datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def fetch_news_via_client(
    query: str,
    start: str,
    end: str,
    max_records: int = 250,
    fetch_bodies: bool = True,
    retries: int = 5,
    _fetch_snippet=None,
    _client_factory=GdeltClient,
) -> list[dict]:
    """Pull news via gdelt-client's DOC API article search.

    Same return contract as the old hand-rolled ``fetch_news``: a list of
    dicts with url, title, body_snippet, published_at, source_domain,
    query, fetched_at. Raises ``RateLimitError`` when the API keeps
    throttling past ``retries`` attempts; callers log that as an explicit
    fetch gap (see ``widen.py`` fetch_log).
    """
    from signal_lab.ingest import _fetch_snippet as _default_snippet

    snippet_fn = _fetch_snippet or _default_snippet
    filters = query_to_filters(query, start, end, max_records)
    client = _client_factory(
        max_retries=retries,
        retry_backoff_base=BACKOFF_BASE_S,
        retry_max_wait=BACKOFF_MAX_S,
    )
    try:
        df = client.article_search(filters)
    except RateLimitError as exc:
        print(f"[gdelt] rate limited past {retries} retries; "
              f"logging as fetch gap: {exc}", file=sys.stderr)
        raise
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return []
    now = datetime.now(timezone.utc)
    rows = df.to_dict("records") if isinstance(df, pd.DataFrame) else []
    metas = [r for r in (_row_to_meta(rw, query, now, "") for rw in rows)
             if r["url"]]
    if fetch_bodies and metas:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as ex:
            snippets = list(ex.map(snippet_fn, [m["url"] for m in metas]))
        for m, s in zip(metas, snippets):
            m["body_snippet"] = s
    return metas
