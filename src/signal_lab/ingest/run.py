"""CLI: python -m signal_lab.ingest.run --tickers AAPL,MSFT --days 90 [--json]

With --json, prints ONLY the JSON summary to stdout; all logs go to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta

from . import DB_PATH, fetch_news, fetch_prices, row_counts, store_news, store_prices

# Company name phrases for GDELT queries, mapped to tickers.
# Finance-context keywords keep results on business news, not fruit recipes.
UNIVERSE = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta",
    "JPM": "JPMorgan",
    "XOM": "Exxon",
    "JNJ": "Johnson & Johnson",
}

QUERY_TEMPLATE = ('("{company} stock" OR "{company} shares" OR "{company} earnings" '
                  'OR "{ticker} stock" OR "{ticker} shares") sourcelang:english')


def _chunk_windows(start: date, end: date, chunk_days: int = 10) -> list[tuple[str, str]]:
    """Split [start, end] into contiguous non-overlapping date windows.

    GDELT returns the most-recent articles first, so a single query over a
    long window clusters everything on the latest days. Chunking spreads
    articles across the whole window, which the downstream weak-labeling
    (next-day returns) and event study need.
    """
    total = (end - start).days
    n = max(2, min(9, total // chunk_days + 1))
    windows = []
    for i in range(n):
        cs = start + timedelta(days=(total * i) // n)
        ce = start + timedelta(days=(total * (i + 1)) // n) if i < n - 1 else end
        windows.append((cs.isoformat(), ce.isoformat()))
    return windows


def run_ingest(tickers: list[str], days: int, max_news: int = 60,
               no_bodies: bool = False, log=print) -> dict:
    """Run the full ingest. Returns a JSON-serializable summary dict."""
    end = date.today()
    start = end - timedelta(days=days)
    s, e = start.isoformat(), end.isoformat()

    log(f"[ingest] prices {tickers} {s}..{e}")
    prices = fetch_prices(tickers, s, e)
    n_p = store_prices(prices)
    log(f"[ingest] prices rows written: {n_p}")

    windows = _chunk_windows(start, end)
    per_chunk = max(10, -(-max_news // len(windows)))  # ceiling division
    log(f"[ingest] news windows: {len(windows)} x {per_chunk} articles per ticker "
        f"({windows[0][0]}..{windows[-1][1]})")

    total_news = 0
    n_queries = 0
    for t in tickers:
        company = UNIVERSE.get(t, t)
        query = QUERY_TEMPLATE.format(company=company, ticker=t)
        for cs, ce in windows:
            if n_queries > 0:
                time.sleep(30)  # GDELT rate limits aggressively; be polite
            n_queries += 1
            log(f"[ingest] news query: {query} [{cs}..{ce}]")
            try:
                items = fetch_news(query, cs, ce, max_records=per_chunk,
                                   fetch_bodies=not no_bodies)
            except Exception as exc:  # one bad query must not kill the run
                log(f"[ingest] query failed for {t} [{cs}..{ce}]: {exc}")
                continue
            n = store_news(items)
            total_news += n
            log(f"[ingest] news rows written for {t} [{cs}..{ce}]: {n}")

    log(f"[ingest] totals: news={total_news} (this run)")
    log(f"[ingest] db={DB_PATH} counts={row_counts()}")
    return {"news_rows": total_news, "price_rows": n_p, "tickers": tickers,
            "start": s, "end": e, "news_windows": len(windows)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest news + prices into DuckDB")
    ap.add_argument("--tickers", default=",".join(UNIVERSE), help="comma-separated tickers")
    ap.add_argument("--days", type=int, default=90, help="lookback window in days")
    ap.add_argument("--max-news", type=int, default=60, help="max articles per company query")
    ap.add_argument("--no-bodies", action="store_true", help="skip article body fetch")
    ap.add_argument("--json", action="store_true",
                    help="print only the JSON summary to stdout (logs to stderr)")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    log = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print
    summary = run_ingest(tickers, args.days, args.max_news, args.no_bodies, log=log)
    if args.json:
        print(json.dumps(sanitize_json(summary)))


if __name__ == "__main__":
    main()
