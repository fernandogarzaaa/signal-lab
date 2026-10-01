"""Workstream 2: historical analogues.

Given a ticker and a date (a news event), find the most similar past news
events for the SAME ticker and show what happened next.

An "event" is a trading day on which scored news was PUBLISHED (each
article belongs to exactly one event: its pub_date, snapped forward to the
next trading day for weekend publication). This matters: if events were
defined by overlapping [date-1d, date+1d] windows, every event's nearest
"analogue" would degenerately be its own neighboring day sharing the same
articles. Per-pub-date assignment keeps analogues distinct episodes.

The query date keeps the [date-1d, date+1d] window convention from
explain_move ("the news around this move"), so the two views tell a
consistent story; candidates are historical news days. Both are compared
as (abnormal_z, sentiment, volume) triples.

Each event is described by three features, all computable with no
lookahead:

  - abnormal_z: the day's 1-day return z-scored against its own trailing
    60 trading days (same definition as explain_move).
  - sentiment: mean article proba mapped to [-1, 1] via p*2-1.
  - volume: log1p(number of articles in the window).

Features are standardized across the ticker's event population, and
analogues are ranked by euclidean distance (lower = more similar). The
query event itself is excluded. For each analogue we report forward
close-to-close returns at +1/+5/+10 trading days, plus a summary of the
median/mean/fraction-positive per horizon.

This is descriptive, not predictive: "on the N most similar past days,
the median 5-day forward return was X%". It must never be presented as a
forecast. The distance metric and per-feature deltas are exposed so the
similarity claim stays auditable.

Loud failures (ValueError): unknown ticker, unparseable date, date outside
price coverage, or a query date with no scored news in its window
(analogues are news-event analogues; without news there is no profile
to match).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

from signal_lab.explain_move import (
    DEFAULT_DB,
    DEFAULT_SCORED,
    _parse_date,
    _price_block,
    _ticker_days,
    load_news_prices,
)

FEATURES = ["abnormal_z", "sentiment", "volume"]
FORWARD_HORIZONS = [1, 5, 10]


def _scored_for_ticker(scored_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    sc = scored_df[scored_df["ticker"] == ticker].copy()
    if sc.empty:
        return sc
    if "pub_date" not in sc.columns:
        sc["pub_date"] = pd.to_datetime(sc["published_at"]).dt.date.astype(str)
    else:
        sc["pub_date"] = sc["pub_date"].astype(str)
    return sc


def _window_articles(sc: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    lo = (day - pd.Timedelta(days=1)).date().isoformat()
    hi = (day + pd.Timedelta(days=1)).date().isoformat()
    return sc[(sc["pub_date"] >= lo) & (sc["pub_date"] <= hi)]


def _title_lookup(news_df: pd.DataFrame) -> dict:
    """url -> title, first wins on duplicates (same rule as explain_move)."""
    titles = {}
    for url, title in zip(news_df["url"].astype(str), news_df["title"].astype(str)):
        if url not in titles:
            titles[url] = title
    return titles


def _top_headline(arts: pd.DataFrame, titles: dict) -> str:
    """Headline of the highest-signal article in the window."""
    if arts.empty:
        return ""
    sig = arts["proba"].clip(0, 1).apply(lambda p: max(p, 1.0 - p))
    url = str(arts.loc[sig.idxmax(), "url"])
    return titles.get(url, "")


def _event_features(day: pd.Timestamp, days: pd.DataFrame,
                    arts: pd.DataFrame, titles: dict) -> dict:
    """Feature vector for one event day. No lookahead: abnormal_z uses only
    trailing prices; sentiment/volume use the articles attached to the day
    (the query's [d-1, d+1] window, or the index event's pub_date group)."""
    pb = _price_block("", day, days, notes=[])
    n = len(arts)
    mean_p = float(arts["proba"].clip(0, 1).mean())
    return {
        "date": day.date().isoformat(),
        "n_articles": n,
        "mean_proba": mean_p,
        "abnormal_z": pb["abnormal_z"],
        "ret_1d": pb["ret_1d"],
        "sentiment": mean_p * 2.0 - 1.0,
        "volume": math.log1p(n),
        "headline": _top_headline(arts, titles),
    }


def build_event_index(ticker: str, days: pd.DataFrame,
                      scored_df: pd.DataFrame, news_df: pd.DataFrame) -> list[dict]:
    """One entry per trading day with scored news PUBLISHED on it.

    Articles are grouped by pub_date (not by overlapping windows), so each
    article belongs to exactly one event. Weekend/holiday pub_dates attach
    to the next trading day. Days whose abnormal_z is unavailable (not
    enough trailing history) are skipped: similarity on a missing feature
    would be dishonest.
    """
    sc = _scored_for_ticker(scored_df, ticker)
    if sc.empty:
        return []
    titles = _title_lookup(news_df)
    trading = set(days["date"])
    buckets: dict = {}
    for pub_date, grp in sc.groupby("pub_date"):
        pd_pub = pd.Timestamp(pub_date).normalize()
        if pd_pub in trading:
            ev_day = pd_pub
        else:
            nxt = days[days["date"] > pd_pub]
            if nxt.empty:
                continue
            ev_day = nxt["date"].iloc[0]
        buckets.setdefault(ev_day, []).append(grp)
    events = []
    for ev_day in sorted(buckets):
        arts = pd.concat(buckets[ev_day])
        feat = _event_features(ev_day, days, arts, titles)
        if feat["abnormal_z"] is None:
            continue
        events.append(feat)
    return events


def _standardize(events: list[dict]) -> tuple[list[dict], dict]:
    """Z-score each feature across the population. Returns (events, stats)
    where each event gains 'z' = {feature: standardized value}."""
    stats = {}
    for f in FEATURES:
        vals = pd.Series([e[f] for e in events], dtype=float)
        mu, sd = float(vals.mean()), float(vals.std(ddof=0))
        stats[f] = {"mean": mu, "std": sd if sd > 0 else 1.0}
    for e in events:
        e["z"] = {f: (e[f] - stats[f]["mean"]) / stats[f]["std"]
                  for f in FEATURES}
    return events, stats


def _forward_returns(day: pd.Timestamp, days: pd.DataFrame) -> dict:
    idx = int(days.index[days["date"] == day][0])
    out = {}
    for h in FORWARD_HORIZONS:
        if idx + h < len(days):
            base = float(days.loc[idx, "close"])
            fwd = float(days.loc[idx + h, "close"])
            out[f"{h}d"] = fwd / base - 1.0
    return out


def _summarize(analogues: list[dict]) -> dict:
    summary = {"n": len(analogues), "horizons": {}}
    for h in FORWARD_HORIZONS:
        key = f"{h}d"
        vals = [a["forward"][key] for a in analogues if key in a["forward"]]
        if not vals:
            continue
        s = pd.Series(vals, dtype=float)
        summary["horizons"][key] = {
            "n": len(vals),
            "median": float(s.median()),
            "mean": float(s.mean()),
            "frac_positive": float((s > 0).mean()),
        }
    return summary


def find_analogues(ticker: str, date, news_df: pd.DataFrame,
                   prices_df: pd.DataFrame, scored_df: pd.DataFrame,
                   k: int = 5) -> dict:
    """Most similar past news events for ticker, plus what happened next."""
    notes: list[str] = []
    ticker = str(ticker).upper().strip()
    known = sorted(pd.unique(prices_df["ticker"]).tolist())
    if ticker not in known:
        raise ValueError(f"unknown ticker {ticker!r}; known: {', '.join(known)}")
    event_day = _parse_date(date)

    days = _ticker_days(prices_df, ticker)
    first, last = days["date"].iloc[0], days["date"].iloc[-1]
    if event_day < first or event_day > last + pd.Timedelta(days=7):
        raise ValueError(
            f"date {event_day.date().isoformat()} outside price coverage "
            f"for {ticker} ({first.date().isoformat()}..{last.date().isoformat()})")

    requested = event_day.date().isoformat()
    if event_day not in set(days["date"]):
        prior = days[days["date"] < event_day]
        if prior.empty:  # pragma: no cover - guarded by the coverage check
            raise ValueError(f"no trading day on or before {requested} for {ticker}")
        event_day = prior["date"].iloc[-1]
        notes.append(
            f"date {requested} is not a trading day; "
            f"using nearest prior trading day {event_day.date().isoformat()}")

    events = build_event_index(ticker, days, scored_df, news_df)
    if not events:
        raise ValueError(f"no scored-news events found for {ticker}")
    query_arts = _window_articles(_scored_for_ticker(scored_df, ticker), event_day)
    if query_arts.empty:
        raise ValueError(
            f"no scored news in window [{(event_day - pd.Timedelta(days=1)).date().isoformat()}, "
            f"{(event_day + pd.Timedelta(days=1)).date().isoformat()}] for {ticker}; "
            f"analogues need a news profile to match")
    query = _event_features(event_day, days, query_arts, _title_lookup(news_df))
    if query["abnormal_z"] is None:
        raise ValueError(
            f"not enough trailing price history before {query['date']} "
            f"to compute an abnormal return")

    events, stats = _standardize(events)
    qz = {f: (query[f] - stats[f]["mean"]) / stats[f]["std"] for f in FEATURES}
    ranked = []
    for e in events:
        if e["date"] == query["date"]:
            continue
        dist = math.sqrt(sum((e["z"][f] - qz[f]) ** 2 for f in FEATURES))
        ranked.append((dist, e))
    ranked.sort(key=lambda t: t[0])

    k = max(1, int(k))
    if k > len(ranked):
        notes.append(
            f"only {len(ranked)} past events available for {ticker}; "
            f"returning all of them")
    picked = ranked[:k]
    analogues = []
    truncated = 0
    for dist, e in picked:
        fwd = _forward_returns(pd.Timestamp(e["date"]), days)
        if len(fwd) < len(FORWARD_HORIZONS):
            truncated += 1
        analogues.append({
            "date": e["date"],
            "distance": float(dist),
            "feature_deltas": {f: float(e["z"][f] - qz[f]) for f in FEATURES},
            "headline": e["headline"],
            "n_articles": e["n_articles"],
            "mean_proba": e["mean_proba"],
            "abnormal_z": e["abnormal_z"],
            "ret_1d": e["ret_1d"],
            "forward": fwd,
        })
    if truncated:
        notes.append(
            f"{truncated} analogue(s) are within {max(FORWARD_HORIZONS)} trading "
            f"days of the end of price coverage; forward returns truncated")
    burst = [a["date"] for a in analogues
             if abs((pd.Timestamp(a["date"]) - event_day).days) <= 3]
    if burst:
        notes.append(
            f"{', '.join(burst)} within 3 days of the query date "
            f"(likely the same news burst, not an independent episode)")

    return {
        "ticker": ticker,
        "requested_date": requested,
        "date": query["date"],
        "query": {
            "headline": query["headline"],
            "n_articles": query["n_articles"],
            "mean_proba": query["mean_proba"],
            "abnormal_z": query["abnormal_z"],
            "ret_1d": query["ret_1d"],
        },
        "analogues": analogues,
        "summary": _summarize(analogues),
        "notes": notes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Historical analogues: similar past news events and what happened next.")
    ap.add_argument("--ticker", required=True, help="e.g. MSFT")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--k", type=int, default=5, help="max analogues (default 5)")
    ap.add_argument("--db", default=None, help="override DuckDB path")
    ap.add_argument("--scored", default=None, help="override scored_news.csv path")
    ap.add_argument("--json", action="store_true",
                    help="print only the JSON result to stdout (logs to stderr)")
    args = ap.parse_args()
    log = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print
    try:
        log(f"[analogues] loading stores for {args.ticker} @ {args.date}")
        news_df, prices_df, scored_df = load_news_prices(args.db, args.scored)
        result = find_analogues(args.ticker, args.date, news_df, prices_df,
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
