"""M1: ingestion. GDELT news + yfinance prices -> DuckDB, idempotent."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
DB_PATH = str(DATA_DIR / "signal_lab.duckdb")

NEWS_SCHEMA = """
CREATE TABLE IF NOT EXISTS news_raw (
    url VARCHAR PRIMARY KEY,
    title VARCHAR,
    body_snippet VARCHAR,
    published_at TIMESTAMP,
    source_domain VARCHAR,
    query VARCHAR,
    fetched_at TIMESTAMP
)
"""
PRICES_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices_daily (
    ticker VARCHAR,
    date DATE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    PRIMARY KEY (ticker, date)
)
"""


def _connect() -> duckdb.DuckDBPyConnection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DB_PATH)
    con.execute(NEWS_SCHEMA)
    con.execute(PRICES_SCHEMA)
    return con


def _upsert(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, keys: list[str]) -> int:
    """Idempotent insert: only brand-new keys are inserted, so a rerun
    changes nothing (and a shared URL keeps its first query attribution).
    Returns the number of rows actually inserted."""
    if df.empty:
        return 0
    cols = list(df.columns)
    col_list = "(" + ", ".join(cols) + ")"
    select_list = ", ".join(f'_new_rows."{c}"' for c in cols)
    cond = " AND ".join(f'{table}."{k}" = _new_rows."{k}"' for k in keys)
    con.register("_new_rows", df)
    before = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    con.execute(
        f"INSERT INTO {table} {col_list} "
        f"SELECT {select_list} FROM _new_rows "
        f"WHERE NOT EXISTS (SELECT 1 FROM {table} WHERE {cond})"
    )
    after = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    con.unregister("_new_rows")
    return after - before


def _fetch_snippet(url: str, timeout: float = 8.0) -> str:
    """Best-effort article body snippet. Never raises; returns '' on failure."""
    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "signal-lab/0.1 (interview-prep research)"},
        )
        if r.status_code != 200 or not r.text:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:600]
    except Exception:
        return ""


def fetch_news(query: str, start: str, end: str, max_records: int = 250,
             fetch_bodies: bool = True, retries: int = 5) -> list[dict]:
    """Pull news via the GDELT 2.1 DOC API.

    start/end: 'YYYY-MM-DD'. Returns list of dicts with url, title,
    body_snippet, published_at, source_domain. Implemented through the
    maintained ``gdelt-client`` package (see
    ``signal_lab.ingest.gdelt_client``); 429s are retried with
    exponential backoff inside the client, and a ``RateLimitError`` is
    raised when retries are exhausted so callers can log an explicit
    fetch gap.

    Code-health migration only: this does not change GDELT's rate
    limits. Politeness sleeps between queries live in the callers.
    """
    from signal_lab.ingest.gdelt_client import fetch_news_via_client
    return fetch_news_via_client(
        query, start, end,
        max_records=max_records, fetch_bodies=fetch_bodies, retries=retries,
    )


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Pull daily OHLCV via yfinance. Returns DataFrame with
    ticker, date, open, high, low, close, volume."""
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=False, progress=False, threads=True
    )
    if raw.empty:
        return pd.DataFrame(
            columns=["ticker", "date", "open", "high", "low", "close", "volume"]
        )
    # yfinance returns MultiIndex columns (field, ticker) for multi-ticker
    frames = []
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    for t in tickers:
        try:
            sub = raw.xs(t, axis=1, level=1) if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        sub = sub.rename(columns=str.lower).reset_index()
        # yfinance index col may be named 'Date' or be DatetimeIndex
        date_col = sub.columns[0]
        sub = sub.rename(columns={date_col: "date"})
        sub["ticker"] = t
        cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
        sub = sub[[c for c in cols if c in sub.columns]]
        sub["date"] = pd.to_datetime(sub["date"]).dt.date
        frames.append(sub)
    if not frames:
        return pd.DataFrame(
            columns=["ticker", "date", "open", "high", "low", "close", "volume"]
        )
    df = pd.concat(frames, ignore_index=True).dropna(subset=["close"])
    return df[["ticker", "date", "open", "high", "low", "close", "volume"]]


def store_news(items: list[dict]) -> int:
    """Upsert into news_raw. Returns rows written. Reruns change nothing."""
    df = pd.DataFrame(items)
    if df.empty:
        return 0
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    df = df[["url", "title", "body_snippet", "published_at", "source_domain", "query", "fetched_at"]]
    con = _connect()
    try:
        return _upsert(con, "news_raw", df, ["url"])
    finally:
        con.close()


def store_prices(df: pd.DataFrame) -> int:
    """Upsert into prices_daily. Returns rows written."""
    if df.empty:
        return 0
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    con = _connect()
    try:
        return _upsert(con, "prices_daily", df, ["ticker", "date"])
    finally:
        con.close()


def row_counts() -> dict:
    con = _connect()
    try:
        return {
            "news_raw": con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0],
            "prices_daily": con.execute("SELECT COUNT(*) FROM prices_daily").fetchone()[0],
        }
    finally:
        con.close()
