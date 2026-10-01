"""Workstream 1: "Why did it move?" query path.

Given a ticker and a date, this module answers the analyst's question with
evidence, not narrative:

  - price: 1-day and 3-day returns around the date, plus a z-score of the
    1-day return against its own trailing 60 trading-day history
    (abnormal_z) and an |z| > 2 outlier flag.
  - drivers: the top-k news articles for that ticker published within
    [date-1d, date+1d], ranked by signal strength max(proba, 1-proba),
    with headlines joined from news_raw (scored_news.csv carries no titles).
  - context: extractive cited 10-K context via signal_lab.rag.explain().
    If the 10-K index is not built, this is a note, not a crash.
  - timeline: trading-day closes for [date-10, date+5] plus driver markers,
    for the dashboard chart.
  - notes: explicit strings for every edge case. A date with no news in
    the window is a VALID answer (drivers=[], plus a note). Never silent.

The logic in explain_move() is pure over DataFrames (no DB, no network)
so tests use synthetic frames. load_news_prices() is the only impure part.

Loud failures (ValueError): unknown ticker, unparseable date, date outside
price coverage entirely.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "data" / "signal_lab.duckdb"
DEFAULT_SCORED = REPO_ROOT / "data" / "artifacts" / "scored_news.csv"

OUTLIER_Z = 2.0
TRAILING_DAYS = 60
MIN_TRAILING = 10


def load_news_prices(db_path: str | Path | None = None,
                     scored_path: str | Path | None = None):
    """Read the real stores: (news_df, prices_df, scored_df).

    news_df: url, title, body_snippet, published_at, source_domain
    prices_df: ticker, date, open, high, low, close, volume
    scored_df: url, text, ticker, published_at, proba
    """
    import duckdb

    db_path = Path(db_path) if db_path else DEFAULT_DB
    scored_path = Path(scored_path) if scored_path else DEFAULT_SCORED
    if not db_path.exists():
        raise ValueError(f"news/prices database not found: {db_path}")
    if not scored_path.exists():
        raise ValueError(f"scored news CSV not found: {scored_path}")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        news_df = con.execute(
            "SELECT url, title, body_snippet, published_at, source_domain "
            "FROM news_raw").fetchdf()
        prices_df = con.execute(
            "SELECT ticker, date, open, high, low, close, volume "
            "FROM prices_daily ORDER BY ticker, date").fetchdf()
    finally:
        con.close()
    scored_df = pd.read_csv(scored_path, parse_dates=["published_at"])
    # Workstream 3: fold in freshly polled monitor scores so the explain
    # and analogue views see new articles. Monitor rows win on URL
    # conflicts; unscored monitor rows are dropped (they need a proba).
    from signal_lab.monitor import merge_scored
    scored_df = merge_scored(scored_df)
    return news_df, prices_df, scored_df


def _parse_date(date) -> pd.Timestamp:
    try:
        ts = pd.to_datetime(date)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"unparseable date: {date!r} (use YYYY-MM-DD)") from exc
    if pd.isna(ts):
        raise ValueError(f"unparseable date: {date!r} (use YYYY-MM-DD)")
    return ts.normalize()


def _ticker_days(prices_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    d = prices_df[prices_df["ticker"] == ticker].copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    return d.sort_values("date").reset_index(drop=True)


def _price_block(ticker: str, event_day: pd.Timestamp,
                 days: pd.DataFrame, notes: list) -> dict:
    """Returns, z-score, outlier flag for the event trading day."""
    idx = int(days.index[days["date"] == event_day][0])
    close = float(days.loc[idx, "close"])
    prev_close = float(days.loc[idx - 1, "close"]) if idx >= 1 else None
    close_3 = float(days.loc[idx - 3, "close"]) if idx >= 3 else None
    ret_1d = (close / prev_close - 1.0) if prev_close else None
    ret_3d = (close / close_3 - 1.0) if close_3 else None
    if prev_close is None:
        notes.append("not enough price history before the event day; 1-day return unavailable")
    if close_3 is None:
        notes.append("fewer than 3 prior trading days; 3-day return unavailable")

    abnormal_z, is_outlier = None, False
    hist = days.iloc[max(0, idx - TRAILING_DAYS):idx]
    if len(hist) >= 2:
        rets = hist["close"].pct_change().dropna()
    else:
        rets = pd.Series(dtype=float)
    if len(rets) >= MIN_TRAILING and rets.std() > 0 and ret_1d is not None:
        abnormal_z = float((ret_1d - rets.mean()) / rets.std())
        is_outlier = abs(abnormal_z) > OUTLIER_Z
    else:
        notes.append(
            f"fewer than {MIN_TRAILING} trailing returns; abnormal z-score unavailable")
    return {
        "event_date": event_day.date().isoformat(),
        "prev_close": prev_close,
        "close": close,
        "ret_1d": ret_1d,
        "ret_3d": ret_3d,
        "abnormal_z": abnormal_z,
        "is_outlier": is_outlier,
    }


def _drivers(ticker: str, event_day: pd.Timestamp, news_df: pd.DataFrame,
             scored_df: pd.DataFrame, k: int, notes: list) -> list[dict]:
    lo = (event_day - pd.Timedelta(days=1)).date().isoformat()
    hi = (event_day + pd.Timedelta(days=1)).date().isoformat()
    sc = scored_df[scored_df["ticker"] == ticker].copy()
    if sc.empty:
        notes.append(f"no scored news for {ticker} at all; drivers unavailable")
        return []
    sc["pub_date"] = pd.to_datetime(sc["published_at"]).dt.date.astype(str)
    win = sc[(sc["pub_date"] >= lo) & (sc["pub_date"] <= hi)].copy()
    if win.empty:
        notes.append(f"no news in window [{lo}, {hi}] for {ticker}")
        return []
    titles = news_df.set_index("url")
    win["signal"] = win["proba"].clip(0, 1).apply(lambda p: max(p, 1.0 - p))
    win = win.sort_values(["signal", "proba"], ascending=[False, False]).head(k)
    out = []
    for _, r in win.iterrows():
        url = str(r["url"])
        title = body = domain = pub = ""
        if url in titles.index:
            meta = titles.loc[url]
            if isinstance(meta, pd.DataFrame):
                meta = meta.iloc[0]
            title = str(meta.get("title") or "")
            body = str(meta.get("body_snippet") or "")
            domain = str(meta.get("source_domain") or "")
            pub = str(meta.get("published_at") or "")
        proba = float(r["proba"])
        out.append({
            "headline": title or "(no headline recorded)",
            "url": url,
            "source_domain": domain,
            "published_at": pub or str(r["published_at"]),
            "pub_date": str(r["pub_date"]),
            "proba": proba,
            "direction": "positive" if proba >= 0.5 else "negative",
            "signal": float(r["signal"]),
            "quote": body[:220],
        })
    return out


def _context(ticker: str, event_date: str, drivers: list, notes: list) -> dict:
    try:
        from signal_lab.rag import explain as rag_explain
    except Exception as exc:  # pragma: no cover - import-time failure
        notes.append(f"10-K context unavailable: rag module failed to import ({exc})")
        return {"note": "10-K index not built"}
    headline = drivers[0]["headline"] if drivers else ""
    try:
        return rag_explain({"ticker": ticker, "event_date": event_date,
                            "headline": headline})
    except Exception as exc:
        notes.append(f"10-K context unavailable: {type(exc).__name__}: {str(exc)[:160]}")
        return {"note": "10-K index not built"}


def _timeline(ticker: str, event_day: pd.Timestamp,
              days: pd.DataFrame, drivers: list) -> dict:
    idx = int(days.index[days["date"] == event_day][0])
    lo, hi = max(0, idx - 10), min(len(days), idx + 6)
    prices = [{"date": d.date().isoformat(), "close": float(c)}
              for d, c in zip(days.loc[lo:hi - 1, "date"], days.loc[lo:hi - 1, "close"])]
    markers = [{"date": d["pub_date"], "headline": d["headline"]} for d in drivers]
    return {"prices": prices, "markers": markers}


def explain_move(ticker: str, date, news_df: pd.DataFrame,
                 prices_df: pd.DataFrame, scored_df: pd.DataFrame,
                 k: int = 5) -> dict:
    """Explain what drove ticker's move on date. Pure over DataFrames."""
    notes: list[str] = []
    ticker = str(ticker).upper().strip()
    known = sorted(pd.unique(prices_df["ticker"]).tolist())
    if ticker not in known:
        raise ValueError(f"unknown ticker {ticker!r}; known: {', '.join(known)}")
    event_day = _parse_date(date)

    days = _ticker_days(prices_df, ticker)
    first, last = days["date"].iloc[0], days["date"].iloc[-1]
    # A date just past the last trading day (weekend/holiday) snaps back to
    # it; anything further out is outside coverage entirely and fails loud.
    if event_day < first or event_day > last + pd.Timedelta(days=7):
        raise ValueError(
            f"date {event_day.date().isoformat()} outside price coverage "
            f"for {ticker} ({first.date().isoformat()}..{last.date().isoformat()})")

    requested = event_day.date().isoformat()
    if event_day not in set(days["date"]):
        prior = days[days["date"] < event_day]
        if prior.empty:  # pragma: no cover - guarded by the coverage check above
            raise ValueError(f"no trading day on or before {requested} for {ticker}")
        event_day = prior["date"].iloc[-1]
        notes.append(
            f"date {requested} is not a trading day; "
            f"using nearest prior trading day {event_day.date().isoformat()}")

    price = _price_block(ticker, event_day, days, notes)
    drivers = _drivers(ticker, event_day, news_df, scored_df, max(1, int(k)), notes)
    context = _context(ticker, price["event_date"], drivers, notes)
    timeline = _timeline(ticker, event_day, days, drivers)
    return {
        "ticker": ticker,
        "requested_date": requested,
        "date": price["event_date"],
        "price": price,
        "drivers": drivers,
        "context": context,
        "timeline": timeline,
        "notes": notes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Why did it move? Evidence for a ticker/date.")
    ap.add_argument("--ticker", required=True, help="e.g. MSFT")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--k", type=int, default=5, help="max driver articles (default 5)")
    ap.add_argument("--db", default=None, help="override DuckDB path")
    ap.add_argument("--scored", default=None, help="override scored_news.csv path")
    ap.add_argument("--json", action="store_true",
                    help="print only the JSON result to stdout (logs to stderr)")
    args = ap.parse_args()
    log = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print
    try:
        log(f"[explain] loading stores for {args.ticker} @ {args.date}")
        news_df, prices_df, scored_df = load_news_prices(args.db, args.scored)
        result = explain_move(args.ticker, args.date, news_df, prices_df,
                              scored_df, k=args.k)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    from signal_lab import sanitize_json
    if args.json:
        print(json.dumps(sanitize_json(result)))
    else:
        print(json.dumps(sanitize_json(result), indent=2))


if __name__ == "__main__":
    main()
