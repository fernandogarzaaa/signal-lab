"""Probability calibration and confidence abstention (0.3.0 WS4).

A classifier that says "70% chance" should be right 70% of the time.
Ours is not, out of the box: class-balanced logistic regression trades
ranking quality for recall, and its raw probabilities are miscalibrated.
This module:

- fits Platt (sigmoid) and isotonic calibrators on a held-out temporal
  calibration split (never the training data, never the test data),
- reports Expected Calibration Error before/after, plus per-bin
  reliability data for the dashboard diagram,
- builds the abstention trade-off: for each confidence threshold tau,
  what fraction of test articles we act on (coverage) and what
  precision/recall/F1 we get on the acted subset.

The honest outcomes we report even when they hurt: calibration that
does not help, or abstention that removes too much coverage to be
useful. A model that knows when not to bet is worth more to a risk
team than a model with a slightly higher F1.

Usage:
    python -m signal_lab.models.calibration [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from signal_lab import sanitize_json
from signal_lab.models import (
    RANDOM_STATE,
    _metrics,
    _take_rows,
    featurize,
    make_text,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ART = REPO_ROOT / "data" / "artifacts"

N_BINS = 10


def expected_calibration_error(y_true, proba, n_bins=N_BINS) -> float:
    """Standard binned ECE: sum_b |acc(b) - conf(b)| * |b| / n."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize puts 1.0 in the last bin; 0.0 in the first.
    bins = np.clip(np.digitize(proba, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        acc = y_true[m].mean()
        conf = proba[m].mean()
        ece += abs(acc - conf) * m.sum() / len(y_true)
    return float(ece)


def reliability_data(y_true, proba, n_bins=N_BINS) -> list[dict]:
    """Per-bin (mean predicted, fraction positive, count) for the diagram."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(proba, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        rows.append(
            {
                "bin": [round(edges[b], 2), round(edges[b + 1], 2)],
                "mean_predicted": round(float(proba[m].mean()), 4) if m.any() else None,
                "fraction_positive": round(float(y_true[m].mean()), 4) if m.any() else None,
                "count": int(m.sum()),
            }
        )
    return rows


def fit_calibrators(y_cal, proba_cal) -> dict:
    """Fit Platt (sigmoid) and isotonic calibrators on held-out proba."""
    y_cal = np.asarray(y_cal, dtype=int)
    proba_cal = np.asarray(proba_cal, dtype=float).reshape(-1, 1)
    platt = LogisticRegression(max_iter=1000)
    platt.fit(proba_cal, y_cal)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(proba_cal.ravel(), y_cal)
    return {"platt": platt, "isotonic": iso}


def apply_calibrators(calibrators, proba) -> dict:
    """Map raw proba through each calibrator."""
    proba = np.asarray(proba, dtype=float)
    return {
        "raw": proba,
        "platt": calibrators["platt"].predict_proba(proba.reshape(-1, 1))[:, 1],
        "isotonic": calibrators["isotonic"].predict(proba),
    }


def abstention_curve(y_true, proba, thresholds=(0.10, 0.15, 0.20, 0.30, 0.50)) -> list[dict]:
    """Coverage vs quality trade-off: act only when P(positive) >= tau.

    This is the trading-relevant form of abstention: we take a position
    only when the model assigns enough probability to the positive
    class. Thresholds span the useful range above the base rate (~0.10).
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    rows = []
    for tau in thresholds:
        keep = proba >= tau
        cov = float(keep.mean())
        if keep.sum() == 0 or len(np.unique(y_true[keep])) < 2:
            rows.append(
                {"tau": tau, "coverage": round(cov, 4), "n_kept": int(keep.sum()),
                 "precision": None, "recall": None, "f1": None,
                 "note": "no positives in kept subset" if keep.sum() else "empty"}
            )
            continue
        rep = _metrics("kept", y_true[keep], proba[keep])
        rows.append(
            {"tau": tau, "coverage": round(cov, 4), "n_kept": int(keep.sum()),
             "precision": round(rep.precision, 4), "recall": round(rep.recall, 4),
             "f1": round(rep.f1, 4)}
        )
    return rows


def run_calibration(df: pd.DataFrame, dense_extra=None, log=print) -> dict:
    """Temporal train (60%) / calibrate (15%) / test (25%) evaluation."""
    work = df.dropna(subset=["published_at", "label"]).reset_index(drop=True)
    if "text" not in work.columns:
        work = work.copy()
        work["text"] = make_text(work)
    texts = work["text"].fillna("").astype(str)
    y = work["label"].astype(int).values

    t = pd.to_datetime(work["published_at"], utc=True)
    order = np.argsort(t.values, kind="stable")
    n = len(work)
    i_cal, i_test = int(n * 0.60), int(n * 0.75)
    tri, cai, tei = order[:i_cal], order[i_cal:i_test], order[i_test:]
    log(f"[cal] temporal split: train={len(tri)} calibrate={len(cai)} test={len(tei)}")

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                          stop_words="english", sublinear_tf=True)
    vec.fit(texts.iloc[tri])
    Xtr = featurize(texts.iloc[tri], vec, dense_extra=_take_rows(dense_extra, tri))
    Xca = featurize(texts.iloc[cai], vec, dense_extra=_take_rows(dense_extra, cai))
    Xte = featurize(texts.iloc[tei], vec, dense_extra=_take_rows(dense_extra, tei))

    clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=RANDOM_STATE)
    clf.fit(Xtr, y[tri])
    calibrators = fit_calibrators(y[cai], clf.predict_proba(Xca)[:, 1])
    proba = apply_calibrators(calibrators, clf.predict_proba(Xte)[:, 1])
    yte = y[tei]

    variants = {}
    for name, p in proba.items():
        rep = _metrics(name, yte, p)
        variants[name] = {
            "ece": round(expected_calibration_error(yte, p), 4),
            "pr_auc": round(rep.pr_auc, 4),
            "f1": round(rep.f1, 4),
            "reliability": reliability_data(yte, p),
        }
        log(f"[cal] {name}: ECE={variants[name]['ece']:.4f} "
            f"PR-AUC={rep.pr_auc:.4f} F1={rep.f1:.4f}")

    # Abstention curves for every variant: the trading question is what a
    # confidence gate does to coverage and precision, calibrated or not.
    best = min(variants, key=lambda k: variants[k]["ece"])
    log(f"[cal] best calibrated: '{best}' (lowest ECE)")
    abstention = {name: abstention_curve(yte, p) for name, p in proba.items()}

    return {
        "split": {"n_train": int(len(tri)), "n_calibrate": int(len(cai)),
                  "n_test": int(len(tei)), "test_pos_rate": round(float(yte.mean()), 4)},
        "variants": variants,
        "best_calibrated": best,
        "abstention": abstention,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration + abstention (0.3.0 WS4)")
    ap.add_argument("--json", action="store_true",
                    help="print only the JSON result to stdout (logs to stderr)")

    from signal_lab.models.build_and_train import build_dataset
    from signal_lab.models.context_features import CONTEXT_FEATURE_NAMES
    from signal_lab.models.run import add_label_args, label_config_from_args

    add_label_args(ap)
    args = ap.parse_args()
    log = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print

    label_cfg = label_config_from_args(args).validated()
    df, _ = build_dataset(log=log, label_cfg=label_cfg)
    result = run_calibration(df, dense_extra=df[CONTEXT_FEATURE_NAMES], log=log)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "calibration.json").write_text(json.dumps(sanitize_json(result), indent=2))
    log(f"[cal] wrote {ART / 'calibration.json'}")
    if args.json:
        print(json.dumps(sanitize_json(result)))


if __name__ == "__main__":
    main()
