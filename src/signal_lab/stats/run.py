"""CLI: python -m signal_lab.stats.run --json

Prints ONLY the JSON result to stdout (logs to stderr):
{"events": [{"ticker":..,"date":..,"abnormal_return":..,"t_stat":..,
             "p_value":..,"ci_low":..,"ci_high":..}, ...],
 "summary": {"n_events":.., "mean_abnormal_return":..,
             "share_significant":.., "window": "[-1,+1]"}}

Per-event abnormal_return is the [-1,+1] CAR. t_stat/p_value/ci are the
cross-sectional window inference (same for every event, documented); the
summary's share_significant is the fraction of events whose CAR falls
outside the window's 95% CI.
"""

from __future__ import annotations

import argparse
import json
import sys

import duckdb
import pandas as pd

from signal_lab.ingest import DB_PATH
from signal_lab import sanitize_json
from signal_lab.stats import event_cars
from signal_lab.stats.run_study import run_study


def run_event_json(log=print) -> dict:
    events, res = run_study(log=log)
    w = res[res.window == "[-1,+1]"].iloc[0] if not res.empty else None

    con = duckdb.connect(DB_PATH, read_only=True)
    prices = con.execute("SELECT ticker, date, close FROM prices_daily").fetchdf()
    con.close()
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    ec = event_cars(events[["ticker", "event_date"]], prices, pre=1, post=1)

    ci_lo = float(w["ci95_lo"]) if w is not None else None
    ci_hi = float(w["ci95_hi"]) if w is not None else None
    ev_list = []
    for _, r in ec.iterrows():
        car = float(r["car"])
        ev_list.append({
            "ticker": r["ticker"],
            "date": str(r["event_date"]),
            "abnormal_return": round(car, 5),
            "t_stat": round(float(w["t_stat"]), 3) if w is not None else None,
            "p_value": round(float(w["p_value"]), 4) if w is not None else None,
            "ci_low": ci_lo, "ci_high": ci_hi,
            "significant": bool(ci_lo is not None and (car < ci_lo or car > ci_hi)),
        })
    cars = ec["car"].values if not ec.empty else []
    sig = [c for c in cars if ci_lo is not None and (c < ci_lo or c > ci_hi)]
    return {
        "events": ev_list,
        "summary": {
            "n_events": len(ev_list),
            "mean_abnormal_return": round(float(cars.mean()), 5) if len(cars) else None,
            "share_significant": round(len(sig) / len(cars), 4) if len(cars) else None,
            "window": "[-1,+1]",
            "window_p_value": round(float(w["p_value"]), 4) if w is not None else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Event study JSON runner")
    ap.add_argument("--json", action="store_true",
                    help="print only the JSON result to stdout (logs to stderr)")
    args = ap.parse_args()
    log = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print
    result = run_event_json(log=log)
    if args.json:
        print(json.dumps(sanitize_json(result)))
    else:
        print(json.dumps(sanitize_json(result), indent=2))


if __name__ == "__main__":
    main()
