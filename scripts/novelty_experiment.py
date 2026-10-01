"""Novelty-feature experiment: does novelty-modulated sentiment beat sentiment
alone out-of-sample?

Arms (all: baseline label scheme, 3 purged walk-forward folds, purge=3d,
embargo=3d, class-balanced logistic regression):

  (a) baseline  = TF-IDF + 8 LM-lexicon + 7 market-context features
                 (the 0.3.0 challenger; reproduces the 0.1412 benchmark)
  (b) +novelty  = (a) + 3 novelty features
                 (nov_novelty, nov_sentiment_x_novelty, nov_abs_sentiment_x_novelty)
  (c) no-lexicon ablation = TF-IDF + 7 context + 3 novelty, WITHOUT the 8
                 LM-lexicon sentiment features (tests the novelty family
                 without standalone sentiment features)
  (d) novelty-only = the 3 novelty features alone (no text at all)

Positioning: commercial news-analytics feeds ship proprietary novelty
metadata; this tests the predictive hypothesis openly -- whether
novelty-modulated sentiment predicts abnormal returns out-of-sample --
with an MIT-licensed, reproducible implementation.

Writes data/artifacts/walk_forward_novelty.json (never touches the
shipped data/artifacts/walk_forward.json) and prints the comparison table.

Usage: .venv/bin/python scripts/novelty_experiment.py [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack as sparse_hstack

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from signal_lab import sanitize_json  # noqa: E402
from signal_lab.models.build_and_train import (  # noqa: E402
    build_dataset,
    load_trading_calendar_days,
)
from signal_lab.models.context_features import CONTEXT_FEATURE_NAMES  # noqa: E402
from signal_lab.models.novelty import NOVELTY_FEATURE_NAMES, build_novelty_features  # noqa: E402
from signal_lab.models.run import baseline_config  # noqa: E402
from signal_lab.models.walk_forward import run_walk_forward  # noqa: E402

ART = REPO_ROOT / "data" / "artifacts"


def _dense_matrix(dense_extra, n_rows: int) -> np.ndarray:
    extra = np.asarray(
        dense_extra.to_numpy(dtype=float)
        if isinstance(dense_extra, pd.DataFrame)
        else dense_extra,
        dtype=float,
    )
    if extra.shape[0] != n_rows:
        raise ValueError(
            f"dense_extra has {extra.shape[0]} rows for {n_rows} texts"
        )
    if np.isnan(extra).any():
        raise ValueError("dense_extra contains NaN; drop or impute first")
    return extra


def featurize_no_lexicon(texts, vectorizer, dense_extra=None):
    """TF-IDF + dense_extra, skipping the 8 LM-lexicon sentiment features."""
    texts = list(texts)
    X_tfidf = vectorizer.transform(texts)
    parts = [X_tfidf]
    if dense_extra is not None:
        parts.append(csr_matrix(_dense_matrix(dense_extra, len(texts))))
    return sparse_hstack(parts, format="csr")


def featurize_dense_only(texts, vectorizer, dense_extra=None):
    """Dense features only; the text/vectorizer are ignored."""
    if dense_extra is None:
        raise ValueError("featurize_dense_only requires dense_extra")
    return csr_matrix(_dense_matrix(dense_extra, len(list(texts))))


def _summarize(result: dict) -> dict:
    folds = result["folds"]
    return {
        "n_folds_scored": len(folds),
        "pr_auc_per_fold": [f["pr_auc"] for f in folds],
        "f1_per_fold": [f["f1"] for f in folds],
        "mean_pr_auc": result["aggregate"]["pr_auc"]["mean"],
        "mean_f1": result["aggregate"]["f1"]["mean"],
        "pr_auc_ci95": result["aggregate"]["pr_auc"]["ci95"],
        "skipped_folds": result["skipped_folds"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Novelty feature experiment")
    ap.add_argument("--json", action="store_true",
                    help="print only the JSON result to stdout")
    args = ap.parse_args()
    log = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print

    label_cfg = baseline_config().validated()
    df, label_block = build_dataset(log=log, label_cfg=label_cfg)
    log(f"[exp] baseline rows: {len(df)}, positive rate: "
        f"{label_block['positive_rate']:.3f}")

    ctx = df[CONTEXT_FEATURE_NAMES]
    nov = build_novelty_features(df, log=log)
    ctx_nov = pd.concat([ctx, nov], axis=1)

    purge = embargo = label_cfg.window_days + 2
    log(f"[exp] folds=3 embargo={embargo}d (purge is exact, per-fold)")
    trading_days = load_trading_calendar_days()

    arms = {}
    arms["(a) baseline"] = run_walk_forward(
        df, dense_extra=ctx, n_splits=3, trading_days=trading_days,
        label_cfg=label_cfg, embargo_days=embargo, log=log,
    )
    arms["(b) +novelty"] = run_walk_forward(
        df, dense_extra=ctx_nov, n_splits=3, trading_days=trading_days,
        label_cfg=label_cfg, embargo_days=embargo, log=log,
    )
    arms["(c) no-lexicon +novelty"] = run_walk_forward(
        df, dense_extra=ctx_nov, n_splits=3, trading_days=trading_days,
        label_cfg=label_cfg, embargo_days=embargo,
        featurize_fn=featurize_no_lexicon, log=log,
    )
    arms["(d) novelty-only"] = run_walk_forward(
        df, dense_extra=nov, n_splits=3, trading_days=trading_days,
        label_cfg=label_cfg, embargo_days=embargo,
        featurize_fn=featurize_dense_only, log=log,
    )

    summary = {name: _summarize(res) for name, res in arms.items()}
    summary["_meta"] = {
        "label_scheme": "baseline",
        "n_splits": 3,
        "purge_days": purge,
        "embargo_days": embargo,
        "novelty_features": NOVELTY_FEATURE_NAMES,
        "note": "novelty computed once over the full frame using only "
                "strictly-prior articles (no leakage); see "
                "signal_lab.models.novelty",
    }
    ART.mkdir(parents=True, exist_ok=True)
    out_path = ART / "walk_forward_novelty.json"
    out_path.write_text(json.dumps(sanitize_json(summary), indent=2))
    log(f"[exp] wrote {out_path}")

    base = summary["(a) baseline"]["mean_pr_auc"]
    lines = ["arm | folds PR-AUC | mean PR-AUC | mean F1 | delta vs (a)"]
    for name, s in summary.items():
        if name.startswith("_"):
            continue
        folds = "/".join(f"{v:.4f}" for v in s["pr_auc_per_fold"])
        delta = s["mean_pr_auc"] - base
        lines.append(
            f"{name} | {folds} | {s['mean_pr_auc']:.4f} "
            f"{s['pr_auc_ci95']} | {s['mean_f1']:.4f} | {delta:+.4f}"
        )
    table = "\n".join(lines)
    log("[exp] comparison\n" + table)

    if args.json:
        print(json.dumps(sanitize_json(summary)))


if __name__ == "__main__":
    main()
