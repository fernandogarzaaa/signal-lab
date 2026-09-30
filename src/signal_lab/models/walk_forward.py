"""Purged walk-forward cross-validation with embargo (0.3.0 WS5).

Evaluates the stage-3 classifier the way a quant research team would:
no random splits on a time series. Each fold trains on history strictly
before its test period, with two decontamination steps after Lopez de
Prado (Advances in Financial Machine Learning, ch. 7):

- PURGE: drop train articles whose label window (t+1..t+window_days
  trading days) overlaps the test period. A label computed from returns
  inside the test window is not knowable at train time.
- EMBARGO: drop train articles published within ``embargo_days`` after
  an earlier test period started. Their pre-publication feature windows
  overlap those test labels' return windows, so keeping them lets test
  information leak into training features.

Per fold we fit the TF-IDF vectorizer on train texts only, train the
challenger (class-balanced logistic regression on text + lexicon +
context features, the production pipeline), and score PR-AUC / F1 /
precision / recall on the held-out test block. Aggregates report mean,
sample std, and 95% t-intervals across folds. A model chosen on one
split is a story; a model stable across five purged folds is evidence.

Usage:
    python -m signal_lab.models.walk_forward [--folds 5] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
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


def walk_forward_splits(
    published_at: pd.Series,
    n_splits: int = 5,
    purge_days: int = 5,
    embargo_days: int = 5,
    min_train: int = 50,
) -> list[dict]:
    """Purged + embargoed walk-forward splits.

    Articles are sorted by publication time and cut into ``n_splits + 1``
    contiguous blocks; block 0 is train-only and blocks 1..n_splits are
    the test folds (expanding window: fold k trains on blocks 0..k-1).
    Splits are positional, so no article ever appears in both train and
    test of the same fold even with duplicate timestamps.

    Returns a list of dicts with keys: fold, train_idx, test_idx (arrays
    of positions into the sorted frame), test_start, test_end.
    """
    t = pd.to_datetime(published_at, utc=True)
    order = np.argsort(t.values, kind="stable")
    t_sorted = t.iloc[order].reset_index(drop=True)
    n = len(t_sorted)
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    # Block boundaries at evenly spaced positions.
    cuts = [int(round(n * k / (n_splits + 1))) for k in range(n_splits + 2)]
    splits = []
    for k in range(1, n_splits + 1):
        # Fold k tests block k = [cuts[k], cuts[k+1]); block 0 is train-only.
        b_k, b_k1 = cuts[k], cuts[k + 1]
        test_start, test_end = t_sorted.iloc[b_k], t_sorted.iloc[b_k1 - 1]
        train_pos = np.arange(0, b_k)
        test_pos = np.arange(b_k, b_k1)

        # PURGE: a train article whose label window reaches into the test
        # period is dropped. purge_days is calendar days covering
        # window_days trading days plus a buffer.
        keep = t_sorted.iloc[train_pos] <= (test_start - pd.Timedelta(days=purge_days))

        # EMBARGO: drop train articles published within embargo_days after
        # any earlier test block started; their feature windows overlap
        # those test labels' return windows.
        for j in range(1, k):
            t_j = t_sorted.iloc[cuts[j]]
            embargoed = (t_sorted.iloc[train_pos] > t_j) & (
                t_sorted.iloc[train_pos] <= t_j + pd.Timedelta(days=embargo_days)
            )
            keep &= ~embargoed

        train_kept = train_pos[keep.values]
        if len(train_kept) < min_train:
            raise ValueError(
                f"fold {k}: only {len(train_kept)} train rows after purge/embargo "
                f"(min_train={min_train}); reduce n_splits or purge/embargo days"
            )
        splits.append(
            {
                "fold": k,
                "train_idx": order[train_kept],
                "test_idx": order[test_pos],
                "test_start": test_start,
                "test_end": test_end,
                "n_purged": int((~keep).sum()),
            }
        )
    return splits


def run_walk_forward(
    df: pd.DataFrame,
    dense_extra=None,
    n_splits: int = 5,
    purge_days: int = 5,
    embargo_days: int = 5,
    min_train: int = 50,
    featurize_fn=None,
    log=print,
) -> dict:
    """Run purged walk-forward CV of the challenger pipeline.

    df needs columns: text (or title), published_at, label.
    dense_extra: optional DataFrame/ndarray aligned with df (context
    features), split positionally per fold like the text matrix.
    featurize_fn: optional ``(texts, vectorizer, dense_extra) -> sparse
    matrix`` overriding the default TF-IDF + lexicon + dense_extra
    construction (used to ablate feature groups, e.g. drop the lexicon).
    Defaults to ``signal_lab.models.featurize``.
    """
    work = df.dropna(subset=["published_at", "label"]).reset_index(drop=True)
    if "text" not in work.columns:
        work = work.copy()
        work["text"] = make_text(work)
    texts = work["text"].fillna("").astype(str)
    y = work["label"].astype(int).values
    featurize_fn = featurize_fn or featurize

    splits = walk_forward_splits(
        work["published_at"],
        n_splits=n_splits,
        purge_days=purge_days,
        embargo_days=embargo_days,
        min_train=min_train,
    )
    log(f"[wf] {n_splits} purged folds (purge={purge_days}d, embargo={embargo_days}d)")

    folds = []
    skipped = []
    for sp in splits:
        tri, tei = sp["train_idx"], sp["test_idx"]
        ytr_all, yte_all = y[tri], y[tei]
        # A fold with one class in train or test cannot be scored: skip it
        # loudly (recorded in the output) rather than crashing or faking
        # a metric. Single-class early folds are a data property, not a bug.
        if len(np.unique(ytr_all)) < 2 or len(np.unique(yte_all)) < 2:
            reason = (
                f"train classes={sorted(map(int, np.unique(ytr_all)))}, "
                f"test classes={sorted(map(int, np.unique(yte_all)))}"
            )
            log(f"[wf] fold {sp['fold']}: SKIPPED ({reason})")
            skipped.append({"fold": sp["fold"], "reason": reason})
            continue
        # Vectorizer fit on train texts only: no vocabulary leakage.
        vec = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        vec.fit(texts.iloc[tri])
        Xtr = featurize_fn(texts.iloc[tri], vec, dense_extra=_take_rows(dense_extra, tri))
        Xte = featurize_fn(texts.iloc[tei], vec, dense_extra=_take_rows(dense_extra, tei))
        ytr, yte = y[tri], y[tei]

        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )
        clf.fit(Xtr, ytr)
        rep = _metrics("logreg_balanced", yte, clf.predict_proba(Xte)[:, 1])
        base_pr_auc = float(yte.mean())  # majority-baseline PR-AUC = pos rate

        folds.append(
            {
                "fold": sp["fold"],
                "test_start": sp["test_start"].isoformat(),
                "test_end": sp["test_end"].isoformat(),
                "n_train": int(len(tri)),
                "n_test": int(len(tei)),
                "n_train_pos": int(ytr.sum()),
                "n_test_pos": int(yte.sum()),
                "n_purged": sp["n_purged"],
                "pr_auc": round(rep.pr_auc, 4),
                "f1": round(rep.f1, 4),
                "precision": round(rep.precision, 4),
                "recall": round(rep.recall, 4),
                "baseline_pr_auc": round(base_pr_auc, 4),
            }
        )
        log(
            f"[wf] fold {sp['fold']}: train={len(tri)} test={len(tei)} "
            f"pos_rate={yte.mean():.3f} pr_auc={rep.pr_auc:.4f} f1={rep.f1:.4f}"
        )

    def _agg(key):
        vals = np.array([f[key] for f in folds], dtype=float)
        if len(vals) == 0:
            raise ValueError(
                "walk-forward: all folds were skipped (single-class train/test); "
                "the dataset is too small or too imbalanced for "
                f"{n_splits} folds"
            )
        vals = np.array([f[key] for f in folds], dtype=float)
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        # 95% CI via Student t (n-1 df); honest about small n.
        half = (
            float(stats.t.ppf(0.975, len(vals) - 1) * sd / np.sqrt(len(vals)))
            if len(vals) > 1 and sd > 0
            else 0.0
        )
        return {
            "mean": round(mean, 4),
            "std": round(sd, 4),
            "ci95": [round(mean - half, 4), round(mean + half, 4)],
        }

    return {
        "n_splits": n_splits,
        "purge_days": purge_days,
        "embargo_days": embargo_days,
        "model": "logreg_balanced",
        "folds": folds,
        "skipped_folds": skipped,
        "aggregate": {
            "pr_auc": _agg("pr_auc"),
            "f1": _agg("f1"),
            "precision": _agg("precision"),
            "recall": _agg("recall"),
            "baseline_pr_auc": _agg("baseline_pr_auc"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Purged walk-forward CV (0.3.0 WS5)")
    ap.add_argument("--folds", type=int, default=5, help="number of test folds")
    ap.add_argument(
        "--repr",
        choices=["tfidf", "finbert_ctx"],
        default="finbert_ctx",
        help="text representation: tfidf (TF-IDF+lexicon, 0.3.0 default) or "
        "finbert_ctx (FinBERT+7 ctx, 0.4.0 default; cache-only, zero-fills "
        "texts without a cached embedding)",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="print only the JSON result to stdout (logs to stderr)",
    )

    # Late import: building the dataset needs the DB and the full pipeline.
    from signal_lab.models.build_and_train import build_dataset
    from signal_lab.models.context_features import CONTEXT_FEATURE_NAMES
    from signal_lab.models.run import add_label_args, label_config_from_args

    add_label_args(ap)
    args = ap.parse_args()
    log = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print

    label_cfg = label_config_from_args(args).validated()
    df, _label_block = build_dataset(log=log, label_cfg=label_cfg)
    ctx = df[CONTEXT_FEATURE_NAMES]
    featurize_fn = None
    if args.repr == "finbert_ctx":
        from signal_lab.models.embeddings import finbert_ctx_matrix

        def featurize_fn(texts, vectorizer, dense_extra=None):
            return finbert_ctx_matrix(texts, dense_extra=dense_extra, log=log)
    result = run_walk_forward(
        df,
        dense_extra=ctx,
        n_splits=args.folds,
        purge_days=label_cfg.window_days + 2,
        embargo_days=label_cfg.window_days + 2,
        featurize_fn=featurize_fn,
        log=log,
    )
    result["representation"] = args.repr
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "walk_forward.json").write_text(json.dumps(sanitize_json(result), indent=2))
    log(f"[wf] wrote {ART / 'walk_forward.json'}")
    if args.json:
        print(json.dumps(sanitize_json(result)))


if __name__ == "__main__":
    main()
