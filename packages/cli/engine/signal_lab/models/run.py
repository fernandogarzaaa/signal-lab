"""CLI: python -m signal_lab.models.run --json

Prints ONLY the JSON comparison to stdout (logs to stderr):
{"baseline": {"pr_auc":..,"f1":..,...},
 "challenger": {...},
 "imbalance_comparison": {"class_weights": {...}, "resampling": {...},
                          "threshold_tuning": {...}},
 "label_config": {"scheme": "windowed", "window_days": 3, ...},
 "feature_importance": {"model": ..., "method": ...,
                        "top_text_terms": [...], "dense_features": [...]}}

Trains on the real pipeline (idempotent: rebuilds dataset + artifacts).

Labeling flags (0.3.0 workstream 1):
  --label-scheme {windowed,baseline}  windowed = cumulative abnormal return
    over t+1..t+window-days + attention filter (default); baseline = the
    0.2.0 t+1 top-decile labeling, kept for comparison.
  --window-days N        forward window for the windowed scheme (default 3)
  --quantile Q           top-quantile cutoff for the positive class (default 0.90)
  --no-attention         disable the attention filter
  --min-articles N       attention filter: minimum distinct articles on a
                         ticker-day (default 2)
  --no-top-quartile      disable the top-quartile OR-branch of the attention
                         filter (keep only the min-articles rule)
  --benchmark TICKER     market benchmark for abnormal returns (default SPY)

Training-label flags (Jev workstream, 2026-10-01):
  --train-labels {weak,jev,jev-conf06}
                         training-label source (default weak = price-derived,
                         reproducible). jev trains on Jev-judged directional
                         labels only (Jev neutrals abstain from training);
                         jev-conf06 additionally drops directionals with
                         confidence < 0.6. Test metrics are always computed
                         against weak labels. Jev sources need the label file
                         (see --jev-labels-path); the default path is
                         data/jev_labels_full.json.
  --jev-labels-path PATH path to the Jev labeling output JSON (default
                         data/jev_labels_full.json, relative to the repo)

Feature flags (0.3.0 workstream 2):
  --finbert              also run the FinBERT ablation (TF-IDF vs FinBERT-only
                         vs combined, same data/split) and add the
                         `finbert_ablation` block to the JSON. Opt-in: torch +
                         transformers stay optional, and the first use
                         downloads ~440MB of model weights. Offline it falls
                         back to TF-IDF+lexicon with a clear log line.
"""

from __future__ import annotations

import argparse
import json
import sys

from signal_lab import sanitize_json
from signal_lab.models.build_and_train import run_pipeline
from signal_lab.models.labels import LabelConfig, baseline_config


def _as_dict(report) -> dict:
    return {
        "precision": round(report.precision, 4),
        "recall": round(report.recall, 4),
        "f1": round(report.f1, 4),
        "pr_auc": round(report.pr_auc, 4),
        "accuracy": round(report.accuracy, 4),
        "threshold": round(report.threshold, 4),
    }


def add_label_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Shared labeling flags; also used by the gold-set sampler."""
    ap.add_argument(
        "--label-scheme",
        choices=["windowed", "baseline"],
        default="windowed",
        help="weak-labeling scheme (default: windowed)",
    )
    ap.add_argument(
        "--window-days",
        type=int,
        default=3,
        help="forward window in trading days for the windowed scheme (default: 3)",
    )
    ap.add_argument(
        "--quantile",
        type=float,
        default=0.90,
        help="top-quantile cutoff for the positive class (default: 0.90)",
    )
    ap.add_argument(
        "--no-attention", action="store_true", help="disable the attention filter"
    )
    ap.add_argument(
        "--min-articles",
        type=int,
        default=2,
        help="attention filter: minimum distinct articles on a ticker-day (default: 2)",
    )
    ap.add_argument(
        "--no-top-quartile",
        action="store_true",
        help="disable the top-quartile OR-branch of the attention filter",
    )
    ap.add_argument(
        "--benchmark",
        default="SPY",
        help="market benchmark ticker for abnormal returns (default: SPY)",
    )
    return ap


def label_config_from_args(args) -> LabelConfig:
    if args.label_scheme == "baseline":
        return baseline_config()
    return LabelConfig(
        scheme="windowed",
        window_days=args.window_days,
        quantile=args.quantile,
        benchmark=args.benchmark,
        attention_enabled=not args.no_attention,
        attention_min_articles=args.min_articles,
        attention_top_quartile=not args.no_top_quartile,
    ).validated()


def run_comparison(
    log=print,
    label_cfg: LabelConfig | None = None,
    finbert: bool = False,
    train_labels: str = "weak",
    jev_labels_path: str | None = None,
) -> dict:
    result = run_pipeline(
        log=log,
        label_cfg=label_cfg,
        finbert=finbert,
        train_labels=train_labels,
        jev_labels_path=jev_labels_path,
    )
    by_name = {r.name: r for r in result["reports"]}
    log(
        f"[m3] best model: {result['best_model']}",
    )
    return {
        "baseline": _as_dict(by_name["logreg_plain"]),
        "challenger": _as_dict(
            by_name.get("lightgbm_balanced", by_name["logreg_balanced"])
        ),
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
        "label_config": result["label_config"],
        "train_labels": result["train_labels"],
        "finbert_ablation": result["finbert_ablation"],
        "feature_importance": result["feature_importance"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train + compare sentiment classifiers")
    ap.add_argument(
        "--json",
        action="store_true",
        help="print only the JSON result to stdout (logs to stderr)",
    )
    ap.add_argument(
        "--finbert",
        action="store_true",
        help="also run the FinBERT ablation (opt-in; downloads ~440MB on "
        "first use, falls back cleanly offline)",
    )
    ap.add_argument(
        "--train-labels",
        choices=["weak", "jev", "jev-conf06"],
        default="weak",
        help="training-label source (default: weak). jev/jev-conf06 train "
        "on Jev-judged labels; test metrics always use weak labels",
    )
    ap.add_argument(
        "--jev-labels-path",
        default=None,
        help="path to the Jev labeling output JSON (default: "
        "data/jev_labels_full.json)",
    )
    add_label_args(ap)
    args = ap.parse_args()
    log = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print
    label_cfg = label_config_from_args(args)
    if args.label_scheme == "baseline" and (
        args.window_days != 3
        or args.no_attention
        or args.no_top_quartile
        or args.min_articles != 2
        or args.quantile != 0.90
        or args.benchmark != "SPY"
    ):
        log(
            "[m3] note: --label-scheme baseline pins the 0.2.0 labeling; "
            "other labeling flags are ignored"
        )
    result = run_comparison(
        log=log,
        label_cfg=label_cfg,
        finbert=args.finbert,
        train_labels=args.train_labels,
        jev_labels_path=args.jev_labels_path,
    )
    if args.json:
        print(json.dumps(sanitize_json(result)))
    else:
        print(json.dumps(sanitize_json(result), indent=2))


if __name__ == "__main__":
    main()
