"""CLI: python -m signal_lab.backtest.run --json

Prints ONLY the JSON result to stdout (logs to stderr):
{"strategies": [{"name":..,"return":..,"sharpe":..,"max_drawdown":..}, ...],
 "significance": {"sharpe_diff_p_value":..,
                  "note": "paired t-test on daily returns, sentiment vs buy-and-hold"}}
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from signal_lab.backtest.run_backtest import run_backtest
from signal_lab import sanitize_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward backtest JSON runner")
    ap.add_argument("--json", action="store_true",
                    help="print only the JSON result to stdout (logs to stderr)")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--hold-days", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    log = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print
    res = run_backtest(args.folds, args.hold_days, args.seed, log=log)

    sent = res[res.strategy == "sentiment_momentum"]
    p_value = (round(float(sent["paired_p"].iloc[0]), 4)
               if not sent.empty and "paired_p" in res.columns
               and pd.notna(sent["paired_p"].iloc[0]) else None)
    strategies = [
        {"name": r["strategy"], "return": r["total_return"], "sharpe": r["sharpe"],
         "max_drawdown": r["max_drawdown"], "n_days": int(r["n_days"]),
         "paired_p": p_value if r["strategy"] == "sentiment_momentum" else None}
        for _, r in res.iterrows()
    ]
    out = {
        "strategies": strategies,
        "significance": {
            "sharpe_diff_p_value": p_value,
            "note": "paired t-test on daily returns, sentiment_momentum vs buy_and_hold",
        },
    }
    if args.json:
        print(json.dumps(sanitize_json(out)))
    else:
        print(json.dumps(sanitize_json(out), indent=2))


if __name__ == "__main__":
    main()
