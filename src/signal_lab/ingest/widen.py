"""Gate-2 data widening: GDELT news + yfinance prices for the S&P 100 universe.

Covers 2023-07-01..2026-06-30 (news, per ticker per quarter) and
2023-07-01..2026-07-15 (prices). Checkpointed and resumable via the
``fetch_log`` table in DuckDB: every (ticker, quarter) is logged as
``ok``/``error`` exactly once it is attempted, so re-running picks up
where a previous run stopped and never repeats completed work.

Gaps are explicit: failed quarters stay ``status='error'`` with the
error message, and :func:`quality_check` fails loudly until every
quarter is ``ok`` and every universe ticker has coverage.

Same conventions as the existing ingest: GDELT DOC 2.1 API, 30s
politeness sleep between queries, exponential backoff on 429 (inside
``fetch_news``), article bodies fetched (title + snippet is the model
text, consistent with the existing rows).

Usage:
    python -m signal_lab.ingest.widen [--tickers AAPL,MSFT] [--check-only]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb

from signal_lab.ingest import (
    DB_PATH,
    fetch_news,
    fetch_prices,
    store_news,
    store_prices,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UNIVERSE = REPO_ROOT / "data" / "universe_gate2.csv"

NEWS_START = date(2023, 7, 1)
NEWS_END = date(2026, 6, 30)
PRICE_START = "2023-07-01"
PRICE_END = "2026-07-15"
MAX_RECORDS = 100
POLITENESS_SLEEP_S = 30

FETCH_LOG_DDL = """
CREATE TABLE IF NOT EXISTS fetch_log (
    ticker VARCHAR NOT NULL,
    quarter_start DATE NOT NULL,
    quarter_end DATE NOT NULL,
    query VARCHAR NOT NULL,
    n_articles INTEGER NOT NULL,
    status VARCHAR NOT NULL,
    error VARCHAR,
    fetched_at TIMESTAMP NOT NULL,
    PRIMARY KEY (ticker, quarter_start)
)
"""


def build_query(company: str, ticker: str) -> str:
    """GDELT query for one ticker. Same template as the existing ingest."""
    return (
        f'("{company} stock" OR "{company} shares" OR "{company} earnings" '
        f'OR "{ticker} stock" OR "{ticker} shares") sourcelang:english'
    )


def load_universe(path: str | Path = DEFAULT_UNIVERSE) -> list[tuple[str, str]]:
    """Read (ticker, company) rows. Fails loudly on any structural problem."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"[widen] universe file not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "ticker" not in rows[0] or "company" not in rows[0]:
        raise ValueError(f"[widen] universe {path} must have ticker,company columns")
    uni = [(r["ticker"].strip().upper(), r["company"].strip()) for r in rows]
    if any(not t or not c for t, c in uni):
        raise ValueError(f"[widen] universe {path} has blank ticker/company rows")
    tickers = [t for t, _ in uni]
    dupes = sorted({t for t in tickers if tickers.count(t) > 1})
    if dupes:
        raise ValueError(f"[widen] universe {path} has duplicate tickers: {dupes}")
    return uni


def quarter_windows(start: date = NEWS_START, end: date = NEWS_END,
                    ) -> list[tuple[date, date]]:
    """Calendar-quarter (start, end) windows covering [start, end]."""
    if start > end:
        raise ValueError(f"[widen] start {start} after end {end}")
    out: list[tuple[date, date]] = []
    # first day of the quarter containing `start`
    qs = date(start.year, 3 * ((start.month - 1) // 3) + 1, 1)
    while qs <= end:
        # first day of next quarter, then step back one day
        nq_month = qs.month + 3
        nq_year = qs.year + (nq_month - 1) // 12
        nq_month = (nq_month - 1) % 12 + 1
        qe = date(nq_year, nq_month, 1) - timedelta(days=1)
        out.append((max(qs, start), min(qe, end)))
        qs = date(nq_year, nq_month, 1)
    return out


def ensure_fetch_log(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(FETCH_LOG_DDL)


def _logged_status(con: duckdb.DuckDBPyConnection) -> dict[tuple[str, str], str]:
    rows = con.execute(
        "SELECT ticker, CAST(quarter_start AS VARCHAR), status FROM fetch_log"
    ).fetchall()
    return {(t, q): s for t, q, s in rows}


def pending_quarters(
    con: duckdb.DuckDBPyConnection,
    universe: list[tuple[str, str]],
    tickers: list[str] | None = None,
) -> list[tuple[str, str, date, date]]:
    """(ticker, company, qstart, qend) not yet logged 'ok'. Errors are retried."""
    want = set(t.upper() for t in tickers) if tickers else None
    logged = _logged_status(con)
    out = []
    for ticker, company in universe:
        if want is not None and ticker not in want:
            continue
        for qs, qe in quarter_windows():
            if logged.get((ticker, str(qs))) == "ok":
                continue
            out.append((ticker, company, qs, qe))
    return out


def _log(con: duckdb.DuckDBPyConnection, ticker: str, qs: date, qe: date,
         query: str, n: int, status: str, error: str | None) -> None:
    con.execute(
        """INSERT OR REPLACE INTO fetch_log
           (ticker, quarter_start, quarter_end, query, n_articles, status, error, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [ticker, qs, qe, query, n, status, error,
         datetime.now(timezone.utc).replace(tzinfo=None)],
    )


def run_widen(
    db_path: str | Path = DB_PATH,
    universe_path: str | Path = DEFAULT_UNIVERSE,
    max_records: int = MAX_RECORDS,
    sleep_s: float = POLITENESS_SLEEP_S,
    tickers: list[str] | None = None,
    fetch_fn=fetch_news,
    log=print,
) -> dict:
    """Fetch all pending (ticker, quarter) news + full price history.

    Returns a summary dict; per-quarter outcomes are checkpointed in
    fetch_log. Failed quarters are logged as errors (gaps), never
    silently skipped.
    """
    universe = load_universe(universe_path)
    con = duckdb.connect(str(db_path))
    try:
        ensure_fetch_log(con)
        # Prices first (one shot; upsert is idempotent).
        all_tickers = [t for t, _ in universe
                       if tickers is None or t in {x.upper() for x in tickers}]
        log(f"[widen] fetching prices for {len(all_tickers)} tickers "
            f"{PRICE_START}..{PRICE_END}")
        px = fetch_prices(all_tickers, PRICE_START, PRICE_END)
        n_px = store_prices(px)
        log(f"[widen] stored {n_px} price rows")

        work = pending_quarters(con, universe, tickers)
        log(f"[widen] {len(work)} ticker-quarters pending")
        ok = err = 0
        for i, (ticker, company, qs, qe) in enumerate(work, 1):
            query = build_query(company, ticker)
            log(f"[widen] [{i}/{len(work)}] {ticker} "
                f"{qs}..{qe}: {query[:60]}...")
            try:
                metas = fetch_fn(query, qs.isoformat(), qe.isoformat(),
                                 max_records=max_records)
                n = store_news(metas)
                _log(con, ticker, qs, qe, query, len(metas), "ok", None)
                ok += 1
                log(f"[widen]   -> {len(metas)} articles, {n} new rows")
            except Exception as exc:  # noqa: BLE001 - gaps must be explicit
                _log(con, ticker, qs, qe, query, 0, "error", f"{type(exc).__name__}: {exc}")
                err += 1
                log(f"[widen]   !! ERROR: {type(exc).__name__}: {exc}")
            if sleep_s and i < len(work):
                time.sleep(sleep_s)
        summary = {
            "universe_tickers": len(all_tickers),
            "quarters_per_ticker": len(quarter_windows()),
            "pending_before": len(work),
            "quarters_ok": ok,
            "quarters_error": err,
            "price_rows_stored": n_px,
        }
        log(f"[widen] done: {ok} ok, {err} errors")
        return summary
    finally:
        con.close()


def quality_check(
    db_path: str | Path = DB_PATH,
    universe_path: str | Path = DEFAULT_UNIVERSE,
) -> dict:
    """Fail loudly unless the widening is complete and sane.

    Raises ValueError listing: quarters not yet ok, quarters with errors,
    tickers with zero fetched articles, tickers missing price coverage.
    """
    universe = load_universe(universe_path)
    tickers = [t for t, _ in universe]
    if not Path(db_path).exists():
        raise ValueError(f"[widen] database not found: {db_path} "
                         f"(run the widener first)")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        problems: list[str] = []
        ensure_tables = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'fetch_log'").fetchone()[0]
        if not ensure_tables:
            raise ValueError("[widen] fetch_log table missing: run the widener first")
        logged = _logged_status(con)
        nq = len(quarter_windows())
        missing = [(t, str(qs)) for t in tickers for qs, _ in quarter_windows()
                   if logged.get((t, str(qs))) != "ok"]
        if missing:
            problems.append(
                f"{len(missing)} ticker-quarters not ok "
                f"(e.g. {missing[:5]})")
        errors = con.execute(
            "SELECT ticker, CAST(quarter_start AS VARCHAR), error FROM fetch_log "
            "WHERE status = 'error'").fetchall()
        if errors:
            problems.append(
                f"{len(errors)} quarters with fetch errors "
                f"(e.g. {errors[:3]}); re-run to retry")
        per_ticker = {
            r[0]: r[1] for r in con.execute(
                "SELECT ticker, SUM(n_articles) FROM fetch_log "
                "WHERE status = 'ok' GROUP BY ticker").fetchall()
        }
        zero = [t for t in tickers if not per_ticker.get(t)]
        if zero:
            problems.append(f"tickers with zero fetched articles: {zero}")
        px = {r[0]: (r[1], r[2]) for r in con.execute(
            "SELECT ticker, COUNT(*), MAX(date) FROM prices_daily "
            "GROUP BY ticker").fetchall()}
        px_missing = [t for t in tickers if t not in px]
        px_stale = [t for t, (_, mx) in px.items()
                    if t in tickers and str(mx) < "2026-07-01"]
        if px_missing:
            problems.append(f"tickers with no price rows: {px_missing}")
        if px_stale:
            problems.append(f"tickers with stale price data (< 2026-07-01): {px_stale}")
        if problems:
            raise ValueError("[widen] QUALITY CHECK FAILED:\n  - "
                             + "\n  - ".join(problems))
        total = con.execute(
            "SELECT SUM(n_articles) FROM fetch_log WHERE status = 'ok'").fetchone()[0]
        return {"per_ticker_articles": per_ticker,
                "total_articles_fetched": int(total or 0),
                "tickers": len(tickers)}
    finally:
        con.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Gate-2 data widening (GDELT + yfinance)")
    ap.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--tickers", default="",
                    help="comma-separated subset (default: whole universe)")
    ap.add_argument("--max-records", type=int, default=MAX_RECORDS)
    ap.add_argument("--sleep", type=float, default=POLITENESS_SLEEP_S)
    ap.add_argument("--check-only", action="store_true",
                    help="run quality_check only")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    import json as _json
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or None
    if args.check_only:
        out = quality_check(args.db, args.universe)
    else:
        out = run_widen(args.db, args.universe, args.max_records, args.sleep, tickers)
    if args.json:
        print(_json.dumps(out, indent=2, default=str))
    else:
        print(out)


if __name__ == "__main__":
    main()
