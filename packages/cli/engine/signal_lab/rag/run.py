"""CLI: python -m signal_lab.rag.run --event '{"ticker":"AAPL","date":"2026-09-15"}' --json

Prints ONLY the JSON explanation to stdout (logs to stderr):
{"text": "...", "citations": [{"chunk_id":..,"quote":..,"source":..}]}

The headline for the event is looked up from the M3 scored_news.csv
artifacts (highest-proba article for that ticker/date) to ground retrieval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from signal_lab.rag import explain
from signal_lab import sanitize_json

REPO_ROOT = Path(__file__).resolve().parents[3]
ART = REPO_ROOT / "data" / "artifacts"
DB = REPO_ROOT / "data" / "signal_lab.duckdb"


def _title_for_url(url: str) -> str:
    """Headline lookup in DuckDB news_raw.

    scored_news.csv carries no title column (it was always read as ""), so
    join on url against news_raw. Missing DB or missing url -> "".
    """
    if not DB.exists():
        return ""
    import duckdb
    try:
        con = duckdb.connect(str(DB), read_only=True)
        try:
            row = con.execute(
                "SELECT title FROM news_raw WHERE url = ?", [url]).fetchone()
        finally:
            con.close()
    except Exception:
        return ""
    return str(row[0] or "") if row else ""


def run_explain(event_json: str, log=print) -> dict:
    event = json.loads(event_json)
    ticker = event["ticker"].upper()
    date = event.get("date") or event.get("event_date")
    headline = event.get("headline", "")
    if not headline:
        scored_path = ART / "scored_news.csv"
        if scored_path.exists():
            scored = pd.read_csv(scored_path, parse_dates=["published_at"])
            scored["pub_date"] = (pd.to_datetime(scored["published_at"], utc=True)
                                    .dt.date.astype(str))
            cand = scored[(scored["ticker"] == ticker) & (scored["pub_date"] == str(date))]
            if not cand.empty:
                top = cand.sort_values("proba", ascending=False).iloc[0]
                headline = _title_for_url(str(top["url"]))
    log(f"[m6] explaining {ticker} @ {date}")
    result = explain({"ticker": ticker, "event_date": date, "headline": headline or ""})
    for c in result["citations"]:
        c["source"] = f"{ticker} 10-K (SEC EDGAR)"
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG explainer JSON runner")
    ap.add_argument("--event", required=True,
                    help='JSON like \'{"ticker":"AAPL","date":"2026-09-15"}\'')
    ap.add_argument("--json", action="store_true",
                    help="print only the JSON result to stdout (logs to stderr)")
    args = ap.parse_args()
    log = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print
    result = run_explain(args.event, log=log)
    if args.json:
        print(json.dumps(sanitize_json(result)))
    else:
        print(json.dumps(sanitize_json(result), indent=2))


if __name__ == "__main__":
    main()
