"""Retry driver for GDELT backfill windows that failed with 429s.

Reads failed (ticker, start, end) triples from /tmp/failed_windows.txt and
re-runs just those queries with a longer politeness gap. Ingest is an
idempotent upsert, so already-stored rows are skipped.
"""
import sys
import time

sys.path.insert(0, "src")
from signal_lab.ingest import fetch_news, store_news
from signal_lab.ingest.run import UNIVERSE, QUERY_TEMPLATE

FAILED_WINDOWS = "/home/hatch/workspace/signal-lab/failed_windows.txt"
PER_CHUNK = 34  # same as the original --days 365 --max-news 300 run
SLEEP_S = 180   # longer politeness gap; GDELT was throttling hard at 120s


def main() -> None:
    pairs = []
    with open(FAILED_WINDOWS) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                pairs.append(tuple(parts))
    print(f"[retry] {len(pairs)} failed windows to retry", flush=True)
    ok, failed = 0, 0
    for i, (ticker, start, end) in enumerate(pairs):
        if i > 0:
            print(f"[retry] politeness sleep {SLEEP_S}s", flush=True)
            time.sleep(SLEEP_S)
        company = UNIVERSE.get(ticker, ticker)
        query = QUERY_TEMPLATE.format(company=company, ticker=ticker)
        print(f"[retry] {ticker} [{start}..{end}] ({i + 1}/{len(pairs)})", flush=True)
        try:
            items = fetch_news(query, start, end, max_records=PER_CHUNK,
                               fetch_bodies=True)
        except Exception as exc:
            print(f"[retry] FAILED {ticker} [{start}..{end}]: {exc}", flush=True)
            failed += 1
            continue
        n = store_news(items)
        print(f"[retry] stored {n} rows for {ticker} [{start}..{end}]", flush=True)
        ok += 1
    print(f"[retry] done: ok={ok} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
