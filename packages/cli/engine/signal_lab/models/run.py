"""CLI: python -m signal_lab.models.run --json

Prints ONLY the JSON comparison to stdout (logs to stderr):
{"baseline": {"pr_auc":..,"f1":..,...},
 "challenger": {...},
 "imbalance_comparison": {"class_weights": {...}, "resampling": {...},
                          "threshold_tuning": {...}}}

Trains on the real pipeline (idempotent: rebuilds dataset + artifacts).
"""

from __future__ import annotations

import argparse
import json
import sys

from signal_lab.models.build_and_train import run_pipeline
from signal_lab import sanitize_json


def _as_dict(report) -> dict:
    return {"precision": round(report.precision, 4),
            "recall": round(report.recall, 4),
            "f1": round(report.f1, 4),
            "pr_auc": round(report.pr_auc, 4),
            "accuracy": round(report.accuracy, 4),
            "threshold": round(report.threshold, 4)}


def run_comparison(log=print) -> dict:
    result = run_pipeline(log=log)
    by_name = {r.name: r for r in result["reports"]}
    log(f"[m3] best model: {result['best_model']}", )
    return {
        "baseline": _as_dict(by_name["logreg_plain"]),
        "challenger": _as_dict(by_name.get("lightgbm_balanced", by_name["logreg_balanced"])),
        "challenger_name": result["best_model"],
        "imbalance_comparison": {
            "class_weights": _as_dict(by_name["logreg_balanced"]),
            "resampling": _as_dict(by_name["logreg_oversampled"]),
            "threshold_tuning": _as_dict(by_name["logreg_balanced_tuned"]),
        },
        "majority_baseline": _as_dict(by_name["majority_baseline"]),
        "positive_rate": round(result["positive_rate"], 4),
        "n_train": result["n_train"],
        "n_test": result["n_test"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train + compare sentiment classifiers")
    ap.add_argument("--json", action="store_true",
                    help="print only the JSON result to stdout (logs to stderr)")
    args = ap.parse_args()
    log = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print
    result = run_comparison(log=log)
    if args.json:
        print(json.dumps(sanitize_json(result)))
    else:
        print(json.dumps(sanitize_json(result), indent=2))


if __name__ == "__main__":
    main()
