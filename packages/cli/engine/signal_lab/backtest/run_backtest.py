"""M5 runner: walk-forward comparison of sentiment vs baselines.

Walk-forward: the date range is split into K sequential folds. In each fold,
the sentiment-spike threshold (p95 of ticker-day mean proba) is computed on
data strictly before the fold, then applied to the fold's dates. No fold ever
sees its own future.

Usage: python -m signal_lab.backtest.run_backtest [--folds 4] [--hold-days 5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from signal_lab.backtest import run
from signal_lab import write_docs_csv
from signal_lab.ingest import DB_PATH

REPO_ROOT = Path(__file__).resolve().parents[3]
ART = REPO_ROOT / "data" / "artifacts"


def expand_events(events: pd.DataFrame, trading_dates: list, hold_days: int) -> pd.DataFrame:
    """Event (ticker, event_date) -> on/off signal rows.

    Emits (ticker, asof=event_date, strength=1.0) and
    (ticker, asof=event_date+hold_days trading days, strength=0.0).
    Overlapping events resolve by latest-asof-wins in the engine.
    """
    rows = []
    tdates = sorted(trading_dates)
    for _, ev in events.iterrows():
        start = int(np.searchsorted(tdates, ev["event_date"]))
        if start >= len(tdates):
            continue
        rows.append({"ticker": ev["ticker"], "asof": tdates[start], "strength": 1.0})
        off = min(start + hold_days, len(tdates) - 1)
        rows.append({"ticker": ev["ticker"], "asof": tdates[off], "strength": 0.0})
    sig = pd.DataFrame(rows, columns=["ticker", "asof", "strength"])
    if sig.empty:
        return sig
    return sig.sort_values(["ticker", "asof", "strength"]).drop_duplicates(
        subset=["ticker", "asof"], keep="last").reset_index(drop=True)


def run_backtest(folds: int = 4, hold_days: int = 5, seed: int = 7,
                 log=print) -> pd.DataFrame:
    """Walk-forward comparison. Returns the per-strategy metrics DataFrame."""

    scored = pd.read_csv(ART / "scored_news.csv", parse_dates=["published_at"])
    scored["pub_date"] = pd.to_datetime(scored["published_at"], utc=True).dt.date
    agg = (scored.groupby(["ticker", "pub_date"])
                 .agg(mean_proba=("proba", "mean"), n=("proba", "size"))
                 .reset_index())

    con = duckdb.connect(DB_PATH, read_only=True)
    prices = con.execute("SELECT ticker, date, close FROM prices_daily").fetchdf()
    con.close()
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    universe = sorted(prices["ticker"].unique())
    all_dates = sorted(prices["date"].unique())

    # Sequential folds over the date range; first chunk is warmup (threshold-only).
    edges = np.array_split(all_dates, folds + 1)
    fold_frames = []
    for k in range(1, len(edges)):
        train_dates = set(d for e in edges[:k] for d in e)
        test_dates = list(edges[k])
        train_agg = agg[agg["pub_date"].isin(train_dates)]
        if train_agg.empty:
            continue
        thresh = train_agg["mean_proba"].quantile(0.95)
        test_agg = agg[agg["pub_date"].isin(test_dates)]
        events = test_agg[(test_agg["mean_proba"] >= thresh) & (test_agg["n"] >= 2)].copy()
        events = events.rename(columns={"pub_date": "event_date"})
        fold_frames.append((test_dates, thresh, events))
        log(f"[m5] fold {k}: train_days={len(train_dates)} test_days={len(test_dates)} "
            f"p95_thresh={thresh:.3f} events={len(events)}")

    sig_sent, sig_bnh, sig_rnd = [], [], []
    rng = np.random.default_rng(seed)
    for test_dates, thresh, events in fold_frames:
        ev = events.copy()
        sig_sent.append(expand_events(ev[["ticker", "event_date"]], all_dates, hold_days))

        first = min(test_dates)
        bnh = pd.DataFrame(
            [{"ticker": t, "asof": first, "strength": 1.0}
             for t in universe if t != "SPY"])
        sig_bnh.append(bnh)

        n_events = len(ev)
        pool = [(t, d) for t in universe if t != "SPY" for d in test_dates]
        picks = rng.choice(len(pool), size=min(n_events, len(pool)), replace=False)
        rnd = pd.DataFrame([{"ticker": pool[i][0], "event_date": pool[i][1]}
                            for i in picks])
        sig_rnd.append(expand_events(rnd, all_dates, hold_days))

    signals = {
        "sentiment_momentum": pd.concat(sig_sent, ignore_index=True) if sig_sent else pd.DataFrame(),
        "buy_and_hold": pd.concat(sig_bnh, ignore_index=True),
        "random_entry": pd.concat(sig_rnd, ignore_index=True) if sig_rnd else pd.DataFrame(),
    }
    for name, s in signals.items():
        log(f"[m5] {name}: {len(s)} signal rows")

    res = run(signals, prices, cost_bps=5.0)
    log(res.to_string(index=False))
    write_docs_csv(res, "m5-results.csv", log)
    return res


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--hold-days", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    run_backtest(args.folds, args.hold_days, args.seed)


if __name__ == "__main__":
    main()
