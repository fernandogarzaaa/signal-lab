"""Workstream 3: near-real-time watchlist monitor on GDELT 15-minute updates.

The monitor polls the GDELT 2.1 DOC API for the user's watchlist tickers,
stores new articles in DuckDB (idempotent upsert), scores them with the
shipped production sentiment model, and records poll health in
``data/artifacts/monitor_state.json``.

Positioning: this is a cited news monitor, not a predictor. It surfaces
fresh articles with their model sentiment score and market context; it
never forecasts returns.

CLI:
    python -m signal_lab.monitor --poll [--tickers MSFT,NVDA]
        [--lookback-min 30] [--max-per-ticker 25] [--json]
    python -m signal_lab.monitor --watchlist-list
    python -m signal_lab.monitor --watchlist-add TSLA
    python -m signal_lab.monitor --watchlist-remove TSLA
    python -m signal_lab.monitor --watchlist-json   # snapshot for the dashboard

Scoring failures are loud: a missing model/vectorizer raises instead of
falling back to zeros. Articles that cannot be scored honestly (e.g.
insufficient pre-publication price history for the context features) are
kept with an empty proba and a score_note, never a fabricated number.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import joblib
import pandas as pd

from signal_lab import sanitize_json
from signal_lab.ingest import (
    DATA_DIR,
    DB_PATH,
    fetch_news,
    fetch_prices,
    store_news,
    store_prices,
)
from signal_lab.ingest.run import QUERY_TEMPLATE, UNIVERSE
from signal_lab.models import featurize, make_text
from signal_lab.models.context_features import CONTEXT_FEATURE_NAMES, build_context_features
from signal_lab.models.lexicon import LEXICON_FEATURE_NAMES
from signal_lab.nlp import extract_baseline

ART = DATA_DIR / "artifacts"
WATCHLIST_PATH = DATA_DIR / "watchlist.json"
MONITOR_STATE_PATH = ART / "monitor_state.json"
MONITOR_SCORED_PATH = ART / "monitor_scored.csv"
SCORED_NEWS_PATH = ART / "scored_news.csv"

# GDELT refreshes roughly every 15 minutes; polling faster than that buys
# nothing. The overlap between polls must exceed the cadence so no article
# falls in a gap (dedupe by URL makes the overlap harmless).
GDELT_CADENCE_MIN = 15
DEFAULT_LOOKBACK_MIN = 30
DEFAULT_MAX_PER_TICKER = 25
POLITENESS_SLEEP_S = 30  # GDELT rate limits aggressively; be polite
BENCHMARK = "SPY"

_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,8}$")


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

def _default_watchlist() -> list[str]:
    return list(UNIVERSE)


def get_watchlist() -> list[str]:
    """Read the watchlist; create it from the default universe on first use."""
    if not WATCHLIST_PATH.exists():
        tickers = _default_watchlist()
        set_watchlist(tickers)
        return tickers
    try:
        data = json.loads(WATCHLIST_PATH.read_text())
        tickers = data.get("tickers", [])
    except (json.JSONDecodeError, AttributeError) as exc:
        raise ValueError(f"watchlist file is corrupt: {WATCHLIST_PATH}: {exc}")
    if not isinstance(tickers, list) or not tickers:
        raise ValueError(f"watchlist file has no tickers: {WATCHLIST_PATH}")
    return [str(t).upper() for t in tickers]


def set_watchlist(tickers: list[str]) -> list[str]:
    """Validate and persist the watchlist. Returns the normalized list."""
    seen: list[str] = []
    for t in tickers:
        t = str(t).strip().upper()
        if not _TICKER_RE.match(t):
            raise ValueError(f"invalid ticker {t!r}: use 1-8 chars A-Z 0-9 . -")
        if t not in seen:
            seen.append(t)
    if not seen:
        raise ValueError("watchlist must contain at least one ticker")
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    WATCHLIST_PATH.write_text(
        json.dumps(
            {"tickers": seen,
             "updated_at": datetime.now(timezone.utc).isoformat()},
            indent=2,
        )
    )
    return seen


def add_ticker(ticker: str) -> list[str]:
    tickers = get_watchlist()
    t = ticker.strip().upper()
    if t not in tickers:
        tickers.append(t)
    return set_watchlist(tickers)


def remove_ticker(ticker: str) -> list[str]:
    tickers = get_watchlist()
    t = ticker.strip().upper()
    if t not in tickers:
        raise ValueError(f"{t} is not on the watchlist")
    tickers.remove(t)
    return set_watchlist(tickers)


# ---------------------------------------------------------------------------
# Scorer (loud failures, no zero-fill)
# ---------------------------------------------------------------------------

def load_scorer():
    """Load the shipped production sentiment scorer.

    Raises FileNotFoundError with an explicit message when the training
    artifacts are absent. There is deliberately no fallback: scoring with a
    made-up number would be worse than failing loudly.
    """
    vec_path = ART / "vectorizer.pkl"
    model_path = ART / "model.pkl"
    name_path = ART / "best_model.txt"
    missing = [p for p in (vec_path, model_path, name_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "sentiment scorer unavailable; missing training artifacts: "
            + ", ".join(str(p) for p in missing)
            + ". Run the model training pipeline first; the monitor will not "
              "score articles with fabricated values."
        )
    vec = joblib.load(vec_path)
    model = joblib.load(model_path)
    name = name_path.read_text().strip()
    return vec, model, name


def _attribute_ticker(text: str, queried: str) -> tuple[str, str]:
    """Attribute an article to a ticker.

    Uses the deterministic alias extractor and takes the highest-confidence
    hit; falls back to the queried ticker when nothing matches (e.g. a
    user-added ticker with no aliases on file). Returns (ticker, method).
    """
    hits = extract_baseline(text)
    if hits:
        best = max(hits, key=lambda h: h[2])
        return best[1], "alias-extract"
    return queried, "query-fallback"


def _prices_for(tickers: list[str], days: int = 120) -> pd.DataFrame:
    """Fetch recent prices for tickers (+benchmark) and upsert into DuckDB."""
    end = date.today()
    start = end - timedelta(days=days)
    all_tickers = list(dict.fromkeys(list(tickers) + [BENCHMARK]))
    df = fetch_prices(all_tickers, start.isoformat(), end.isoformat())
    n = store_prices(df)
    return df, n


def _read_prices(tickers: list[str]) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        q = "SELECT ticker, date, close FROM prices_daily WHERE ticker IN (%s)" % (
            ",".join("?" for _ in tickers)
        )
        return con.execute(q, tickers).fetchdf()
    finally:
        con.close()


def score_new_articles(items: list[dict], queried: str, log=print) -> pd.DataFrame:
    """Score freshly fetched GDELT items with the production model.

    Returns a DataFrame with columns
    url, text, ticker, published_at, pub_date, proba, score_note.
    Rows that cannot be scored honestly keep proba empty and carry a note.
    """
    rows = []
    for it in items:
        title = (it.get("title") or "").strip()
        if not title:
            continue  # same rule as the training pipeline: no title, no row
        text = (title + " " + (it.get("body_snippet") or "")).strip()
        ticker, method = _attribute_ticker(text, queried)
        pub = it.get("published_at")
        rows.append(
            {
                "url": it.get("url"),
                "text": text,
                "ticker": ticker,
                "attrib_method": method,
                "published_at": pub,
                "pub_date": pd.to_datetime(pub, utc=True).date()
                if pub is not None
                else None,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df.assign(proba=pd.Series(dtype=float), score_note="")

    vec, model, name = load_scorer()
    log(f"[monitor] scoring {len(df)} articles with {name}")

    # The feature space must match the artifact the model was trained on.
    # featurize() always yields TF-IDF (5000) + 8 lexicon features; the 7
    # market-context columns are only appended when the loaded model was
    # trained with them. Anything else is a loud error, not a guess.
    n_base = len(vec.vocabulary_) + len(LEXICON_FEATURE_NAMES)
    n_expected = int(getattr(model, "n_features_in_", n_base))
    use_ctx = n_expected == n_base + len(CONTEXT_FEATURE_NAMES)
    if n_expected != n_base and not use_ctx:
        raise ValueError(
            f"model {name!r} expects {n_expected} features but the "
            f"TF-IDF+lexicon base is {n_base} (context block is "
            f"{len(CONTEXT_FEATURE_NAMES)}): unknown feature space; "
            f"retrain the model or fix the artifact"
        )
    if not use_ctx:
        log(f"[monitor] model trained without context features "
            f"({n_expected} feats); scoring on TF-IDF+lexicon only")

    prices = _read_prices(sorted(set(df["ticker"]) | {BENCHMARK}))
    ctx = build_context_features(df, prices, benchmark=BENCHMARK, log=log)
    ctx_ok = ~ctx[CONTEXT_FEATURE_NAMES].isna().any(axis=1)

    proba = pd.Series(float("nan"), index=df.index)
    note = pd.Series("", index=df.index)
    if ctx_ok.any():
        dense = ctx.loc[ctx_ok, CONTEXT_FEATURE_NAMES] if use_ctx else None
        X = featurize(
            make_text(df[ctx_ok]),
            vec,
            dense_extra=dense,
        )
        proba.loc[ctx_ok] = model.predict_proba(X)[:, 1]
    if not use_ctx:
        note.loc[ctx_ok] = "scored without context features (model trained pre-ctx)"
    note.loc[~ctx_ok] = "unscored: insufficient pre-publication price history"
    log(
        f"[monitor] scored {int(ctx_ok.sum())}/{len(df)} "
        f"({int((~ctx_ok).sum())} unscored)"
    )
    df["proba"] = proba
    df["score_note"] = note
    return df[["url", "text", "ticker", "published_at", "pub_date", "proba", "score_note"]]


def append_monitor_scored(df: pd.DataFrame) -> int:
    """Append scored rows to monitor_scored.csv, deduped by URL. Returns new rows."""
    if df.empty:
        return 0
    ART.mkdir(parents=True, exist_ok=True)
    if MONITOR_SCORED_PATH.exists():
        old = pd.read_csv(MONITOR_SCORED_PATH)
        df = df[~df["url"].isin(set(old["url"]))]
    else:
        df.to_csv(MONITOR_SCORED_PATH, index=False)
        return len(df)
    if df.empty:
        return 0
    df.to_csv(MONITOR_SCORED_PATH, mode="a", header=False, index=False)
    return len(df)


def merge_scored(base_df: pd.DataFrame) -> pd.DataFrame:
    """Fold freshly polled monitor scores into a scored-news frame.

    Monitor rows win on URL conflicts. Unscored monitor rows (empty proba)
    are dropped: downstream consumers need a real number. The extra
    ``score_note`` column is harmless to consumers that ignore it.
    """
    if MONITOR_SCORED_PATH.exists():
        mdf = pd.read_csv(MONITOR_SCORED_PATH, parse_dates=["published_at"])
        mdf = mdf.dropna(subset=["proba"])
        base_df = pd.concat([base_df, mdf], ignore_index=True)
    df = base_df.dropna(subset=["proba"])
    df = df.sort_values("published_at").drop_duplicates("url", keep="last")
    return df.reset_index(drop=True)


def load_scored_news() -> pd.DataFrame:
    """Training-artifact scores plus monitor scores, monitor wins on URL conflicts.

    Unscored monitor rows (empty proba) are dropped: downstream consumers
    need a real number, and the watchlist view reads them separately.
    """
    if not SCORED_NEWS_PATH.exists():
        return pd.DataFrame()
    base = pd.read_csv(SCORED_NEWS_PATH, parse_dates=["published_at"])
    return merge_scored(base)


# ---------------------------------------------------------------------------
# Poll
# ---------------------------------------------------------------------------

def _write_state(state: dict) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    MONITOR_STATE_PATH.write_text(json.dumps(sanitize_json(state), indent=2))


def read_state() -> dict:
    if not MONITOR_STATE_PATH.exists():
        return {"status": "never-polled", "polled_at": None, "tickers": {}}
    return json.loads(MONITOR_STATE_PATH.read_text())


def poll_once(
    tickers: list[str] | None = None,
    lookback_min: int = DEFAULT_LOOKBACK_MIN,
    max_per_ticker: int = DEFAULT_MAX_PER_TICKER,
    fetch_bodies: bool = True,
    sleep_s: float = POLITENESS_SLEEP_S,
    log=print,
) -> dict:
    """Run one poll cycle over the watchlist. Never raises on a ticker
    failure: per-ticker errors are recorded in the state file and the poll
    continues. Returns a JSON-serializable summary."""
    tickers = tickers or get_watchlist()
    started = datetime.now(timezone.utc)
    since = started - timedelta(minutes=lookback_min)
    today = started.date().isoformat()
    log(f"[monitor] poll start {started.isoformat()} tickers={tickers} "
        f"lookback_min={lookback_min}")

    # Refresh price context first (also feeds the watchlist price badges).
    try:
        _, n_price_rows = _prices_for(tickers)
        price_error = None
    except Exception as exc:
        n_price_rows, price_error = 0, f"{type(exc).__name__}: {exc}"
        log(f"[monitor] price fetch failed: {price_error}")

    per_ticker: dict[str, dict] = {}
    total_new = total_scored = 0
    for i, t in enumerate(tickers):
        company = UNIVERSE.get(t, t)
        query = QUERY_TEMPLATE.format(company=company, ticker=t)
        rec: dict = {"status": "ok", "articles_seen": 0, "articles_new": 0,
                     "articles_scored": 0}
        try:
            items = fetch_news(query, today, today, max_records=max_per_ticker,
                               fetch_bodies=fetch_bodies)
            # fetch_news only takes day granularity; keep the trailing window.
            recent = []
            for it in items:
                pub = it.get("published_at")
                if pub is None:
                    continue
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
                if pub >= since:
                    recent.append(it)
            items = recent
            rec["articles_seen"] = len(items)
            n_new = store_news(items)
            rec["articles_new"] = n_new
            if items:
                scored = score_new_articles(items, t, log=log)
                n_scored_rows = append_monitor_scored(scored)
                rec["articles_scored"] = int(scored["proba"].notna().sum())
                rec["rows_appended"] = n_scored_rows
            total_new += n_new
            total_scored += rec["articles_scored"]
        except Exception as exc:  # one bad ticker must not kill the poll
            rec["status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            log(f"[monitor] ticker {t} failed: {rec['error']}")
        per_ticker[t] = rec
        if i < len(tickers) - 1 and sleep_s > 0:
            time.sleep(sleep_s)

    finished = datetime.now(timezone.utc)
    state = {
        "status": "ok" if all(r["status"] == "ok" for r in per_ticker.values())
        else "degraded",
        "polled_at": finished.isoformat(),
        "duration_s": round((finished - started).total_seconds(), 1),
        "lookback_min": lookback_min,
        "tickers": per_ticker,
        "totals": {"articles_new": total_new, "articles_scored": total_scored},
        "price_rows_written": n_price_rows,
        "price_error": price_error,
        "gdelt_note": (
            f"GDELT DOC API refreshes roughly every {GDELT_CADENCE_MIN} minutes; "
            "polling faster than that buys nothing."
        ),
    }
    _write_state(state)
    log(f"[monitor] poll done: status={state['status']} new={total_new} "
        f"scored={total_scored}")
    return state


# ---------------------------------------------------------------------------
# Dashboard snapshot
# ---------------------------------------------------------------------------

def watchlist_snapshot(max_articles_per_ticker: int = 8) -> dict:
    """Everything the dashboard watchlist view needs in one dict."""
    tickers = get_watchlist()
    state = read_state()

    prices: dict[str, dict] = {}
    pdf = _read_prices(tickers)
    if not pdf.empty:
        pdf["date"] = pd.to_datetime(pdf["date"])
        for t, sub in pdf.groupby("ticker"):
            sub = sub.sort_values("date")
            last = sub.iloc[-1]
            prev = sub.iloc[-2] if len(sub) > 1 else None
            chg = (float(last["close"] / prev["close"]) - 1.0) * 100.0 if prev is not None else None
            prices[t] = {
                "last_close": round(float(last["close"]), 2),
                "as_of": str(last["date"].date()),
                "change_pct": round(chg, 2) if chg is not None else None,
            }

    articles: dict[str, list[dict]] = {t: [] for t in tickers}
    if MONITOR_SCORED_PATH.exists():
        mdf = pd.read_csv(MONITOR_SCORED_PATH, parse_dates=["published_at"])
        mdf = mdf.sort_values("published_at", ascending=False)
        for t in tickers:
            sub = mdf[mdf["ticker"] == t].head(max_articles_per_ticker)
            for _, r in sub.iterrows():
                articles[t].append(
                    {
                        "url": r["url"],
                        "title": (r["text"] or "")[:220],
                        "published_at": str(r["published_at"]),
                        "proba": None if pd.isna(r["proba"]) else round(float(r["proba"]), 4),
                        "score_note": r.get("score_note") or "",
                    }
                )

    return sanitize_json(
        {
            "tickers": tickers,
            "prices": prices,
            "articles": articles,
            "monitor": state,
        }
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Signal Lab watchlist monitor")
    ap.add_argument("--poll", action="store_true", help="run one poll cycle")
    ap.add_argument("--tickers", default=None,
                    help="comma-separated tickers (default: watchlist)")
    ap.add_argument("--lookback-min", type=int, default=DEFAULT_LOOKBACK_MIN,
                    help="trailing window in minutes")
    ap.add_argument("--max-per-ticker", type=int, default=DEFAULT_MAX_PER_TICKER)
    ap.add_argument("--no-bodies", action="store_true",
                    help="skip article body fetch (title-only scoring)")
    ap.add_argument("--sleep-s", type=float, default=POLITENESS_SLEEP_S,
                    help="politeness sleep between ticker queries")
    ap.add_argument("--watchlist-list", action="store_true")
    ap.add_argument("--watchlist-add", default=None, metavar="TICKER")
    ap.add_argument("--watchlist-remove", default=None, metavar="TICKER")
    ap.add_argument("--watchlist-json", action="store_true",
                    help="print the dashboard snapshot")
    ap.add_argument("--json", action="store_true",
                    help="print only the JSON result to stdout (logs to stderr)")
    args = ap.parse_args()
    log = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print

    try:
        if args.watchlist_add:
            result = {"tickers": add_ticker(args.watchlist_add)}
        elif args.watchlist_remove:
            result = {"tickers": remove_ticker(args.watchlist_remove)}
        elif args.watchlist_list or (not args.poll and not args.watchlist_json):
            result = {"tickers": get_watchlist()}
        elif args.watchlist_json:
            result = watchlist_snapshot()
        elif args.poll:
            tickers = (
                [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
                if args.tickers
                else None
            )
            result = poll_once(
                tickers,
                lookback_min=args.lookback_min,
                max_per_ticker=args.max_per_ticker,
                fetch_bodies=not args.no_bodies,
                sleep_s=args.sleep_s,
                log=log,
            )
        print(json.dumps(sanitize_json(result)))
    except Exception as exc:
        # Loud failure: the CLI never prints a fabricated empty success.
        err = {"error": f"{type(exc).__name__}: {exc}"}
        if args.json:
            print(json.dumps(err))
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
