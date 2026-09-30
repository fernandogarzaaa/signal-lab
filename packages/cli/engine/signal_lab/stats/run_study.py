"""Run the M4 event study.

Events: ticker-days whose mean predicted sentiment probability is at or
above the 95th percentile (and backed by >= 2 articles). Uses the M3
scored_news.csv artifacts.

Usage: python -m signal_lab.stats.run_study
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from signal_lab.ingest import DB_PATH
from signal_lab import write_docs_csv
from signal_lab.stats import event_study

REPO_ROOT = Path(__file__).resolve().parents[3]
ART = REPO_ROOT / "data" / "artifacts"


def run_study(log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (events, window_results). events has per-event CAR for [-1,+1]."""
    scored = pd.read_csv(ART / "scored_news.csv", parse_dates=["published_at"])
    scored["pub_date"] = pd.to_datetime(scored["published_at"], utc=True).dt.date
    agg = (scored.groupby(["ticker", "pub_date"])
                 .agg(mean_proba=("proba", "mean"), n=("proba", "size"))
                 .reset_index())
    cutoff = agg["mean_proba"].quantile(0.95)
    events = agg[(agg["mean_proba"] >= cutoff) & (agg["n"] >= 2)].copy()
    events = events.rename(columns={"pub_date": "event_date"})
    events["sentiment"] = events["mean_proba"]
    log(f"[m4] ticker-days: {len(agg)}, spike cutoff p95: {cutoff:.3f}, "
        f"events: {len(events)}")

    con = duckdb.connect(DB_PATH, read_only=True)
    prices = con.execute("SELECT ticker, date, close FROM prices_daily").fetchdf()
    con.close()
    prices["date"] = pd.to_datetime(prices["date"]).dt.date

    res = event_study(events[["ticker", "event_date", "sentiment"]], prices)
    log(res.to_string(index=False))
    write_docs_csv(res, "m4-results.csv", log)

    for _, r in res.iterrows():
        verdict = "REJECT H0" if r["p_value"] < 0.05 else "fail to reject H0"
        log(f"[m4] window {r['window']}: n={r['n']} mean CAR={r['mean_car']:+.4f} "
            f"t={r['t_stat']:.2f} p={r['p_value']:.4f} -> {verdict}")
    return events, res


def main() -> None:
    run_study()


if __name__ == "__main__":
    main()
