"""Purged walk-forward cross-validation with exact [t0, t1] decontamination.

0.6.0 (Phase 1) rewrite. The validation logic now lives in
``signal_lab.validation``; this module is the modeling runner on top of it:

- Labels are anchored at t0 (first tradable time after publication), not
  the publication calendar date.
- The label cutoff and the attention-filter quantiles are computed on
  train data only, per fold (audit findings 4c/4d).
- Purge is exact: a train row is dropped when its [t0, t1] interval
  overlaps ANY test range up to and including the current fold.
- Embargo is in trading days on the union calendar.
- Same-ticker same-t0 articles are deduped or down-weighted
  (default: sample_weight = 1/group size).
- Walk-forward defaults to the frozen DEVELOPMENT period
  (t0 <= 2026-06-30); --include-test / --final touch the untouched test
  period and are reserved for the Phase 3 gate.

``walk_forward_splits`` (calendar-day purge approximation) is kept for
the legacy single-split ``train()`` path only; no validation path uses it.

Usage:
    python -m signal_lab.models.walk_forward [--folds 5] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

from signal_lab import sanitize_json
from signal_lab.models import (
    RANDOM_STATE,
    _metrics,
    _take_rows,
    featurize,
    fit_challenger_candidates,
    make_text,
)
from signal_lab.models.labels import LabelConfig
from signal_lab.validation import dedupe, labels as vlabels, periods, point_in_time, splits

REPO_ROOT = Path(__file__).resolve().parents[3]
ART = REPO_ROOT / "data" / "artifacts"

MODEL_CHOICES = (
    "logreg_plain",
    "logreg_balanced",
    "logreg_oversampled",
    "logreg_balanced_tuned",
    "lightgbm_balanced",
)


def walk_forward_splits(
    published_at: pd.Series,
    n_splits: int = 5,
    purge_days: int = 5,
    embargo_days: int = 5,
    min_train: int = 50,
) -> list[dict]:
    """Purged + embargoed walk-forward splits.

    LEGACY (audit: calendar-day purge approximation). Kept for the
    single-split ``train()`` path only; the validation framework in
    ``signal_lab.validation.splits`` replaces it everywhere else.

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
    cuts = [int(round(n * k / (n_splits + 1))) for k in range(n_splits + 1 + 1)]
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


def _resolve_model(fitted: dict, model_name: str) -> object:
    """Pick the requested candidate; logreg_balanced_tuned is the balanced
    estimator (threshold tuning is a reporting detail, not a fit)."""
    key = "logreg_balanced" if model_name == "logreg_balanced_tuned" else model_name
    if key not in fitted:
        raise ValueError(
            f"model {model_name!r} could not be fitted "
            f"(available: {sorted(fitted)}); refusing to substitute silently"
        )
    return fitted[key]


def _fold_train_labels(
    work: pd.DataFrame,
    tri_k: np.ndarray,
    cutoff: float,
    train_label_col: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """(train_positions, y_train) for one fold's training block.

    Weak labels come from the per-fold cutoff; an alternate
    ``train_label_col`` (e.g. Jev labels) replaces them, with NaN
    abstentions excluded. Returns positions into ``work``.
    """
    if train_label_col is not None:
        if train_label_col not in work.columns:
            raise ValueError(f"train_label_col={train_label_col!r} not in df columns")
        all_tr = work[train_label_col].to_numpy(dtype=float)
        keep = ~np.isnan(all_tr[tri_k])
        pos = tri_k[keep]
        return pos, all_tr[pos].astype(int)
    y_all = vlabels.fold_labels(work["abn_ret"], cutoff).to_numpy()
    return tri_k, y_all[tri_k]


def run_walk_forward(
    df: pd.DataFrame,
    dense_extra=None,
    n_splits: int = 5,
    trading_days: np.ndarray | None = None,
    label_cfg: LabelConfig | None = None,
    window: str = "expanding",
    train_window_days: int = 365,
    embargo_days: int = 5,
    purge: bool = True,
    min_train: int = 50,
    retrain_every: int = 1,
    dev_end: date | None = None,
    dedupe_policy: str = "weight",
    include_test: bool = False,
    model_name: str = "logreg_balanced",
    featurize_fn=None,
    log=print,
    train_label_col: str | None = None,
    train_label_source: str | None = None,
    return_oos: bool = False,
) -> dict:
    """Purged walk-forward CV on the validation framework (0.6.0).

    df needs columns: text (or title), published_at, ticker, pub_date,
    t0, t1, abn_ret, ctx_asof. dense_extra: optional DataFrame/ndarray
    aligned with df (context features), split positionally per fold like
    the text matrix. featurize_fn: optional
    ``(texts, vectorizer, dense_extra) -> matrix`` overriding the default
    TF-IDF + lexicon + dense_extra construction.
    trading_days: sorted unique trading-date ordinals (union calendar),
    required for embargo arithmetic.
    train_label_col: optional column of alternate training labels; test
    folds are always scored against the per-fold weak labels, so the
    evaluation target never moves.
    purge: exact purge of train rows whose [t0, t1] reaches into the test
    period. Default True. Set False ONLY for the leakage canary
    (test_leakage.py), which proves the canary is caught with purge=True
    and escapes with purge=False.

    Per fold: per-fold attention (train-only quantiles), per-fold cutoff,
    dedupe/down-weight, TF-IDF fit on train texts only, fit the
    ``model_name`` challenger, score PR-AUC / F1 / precision / recall on
    the held-out test block. With ``retrain_every=r`` the fit is reused
    for r-1 folds (splits and labels are identical either way).
    With ``return_oos=True`` the result also carries ``oos_proba``:
    {work_position: proba} for every fold-test row (out-of-sample scores
    for the scored_news.csv rework).
    """
    label_cfg = (label_cfg or LabelConfig()).validated()
    if trading_days is None:
        raise ValueError("trading_days (union calendar ordinals) is required")
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"model_name must be one of {MODEL_CHOICES}, got {model_name!r}")

    work = df.dropna(subset=["published_at", "t0", "t1", "abn_ret"]).reset_index(drop=True)
    if "text" not in work.columns:
        work = work.copy()
        work["text"] = make_text(work)
    point_in_time.check_frame(work, "run_walk_forward")

    eff_dev_end = periods.TEST_END if include_test else (dev_end or periods.DEV_END)
    if eff_dev_end > periods.DEV_END:
        log(
            "[wf] WARNING: dev_end "
            f"{eff_dev_end.isoformat()} is past the frozen development end "
            f"{periods.DEV_END.isoformat()}: the untouched TEST period is in "
            "the folds. Reserved for the Phase 3 gate; never for selection."
        )
    # Filter to the effective development range BEFORE make_splits: its
    # check_no_confirmation loudly rejects confirmation rows, so they must
    # be excluded here (not silently, the filter is explicit and logged).
    n_before = len(work)
    work = work[
        pd.to_datetime(work["t0"]).dt.date <= eff_dev_end
    ].reset_index(drop=True)
    log(f"[wf] {n_before - len(work)} rows past {eff_dev_end.isoformat()} excluded")
    config = splits.WalkForwardConfig(
        n_splits=n_splits,
        window=window,
        train_window_days=train_window_days,
        horizon_days=label_cfg.window_days,
        embargo_days=embargo_days,
        purge=purge,  # escape hatch for the leakage canary ONLY
        min_train=min_train,
        retrain_every=retrain_every,
        dev_end=eff_dev_end,
    )
    wf_splits = splits.make_splits(work, config, np.asarray(trading_days))
    log(
        f"[wf] {n_splits} purged folds ({window}, horizon={label_cfg.window_days}d, "
        f"embargo={embargo_days}d, dedupe={dedupe_policy}, model={model_name})"
    )

    texts = work["text"].fillna("").astype(str)
    featurize_fn = featurize_fn or featurize
    folds, skipped = [], []
    oos: dict[int, float] = {}
    last_fit: tuple | None = None

    for sp in wf_splits:
        tri, tei = sp["train_idx"], sp["test_idx"]
        fold_pos = np.concatenate([tri, tei])
        fold_df = work.iloc[fold_pos]
        train_df = work.iloc[tri]
        # Per-fold attention: quantiles estimated on the train block only.
        if label_cfg.attention_enabled:
            mask = vlabels.attention_keep_mask(
                train_df,
                fold_df,
                min_articles=label_cfg.attention_min_articles,
                top_quartile=label_cfg.attention_top_quartile,
            ).to_numpy()
        else:
            mask = np.ones(len(fold_df), dtype=bool)
        tri_k, tei_k = tri[mask[: len(tri)]], tei[mask[len(tri):]]
        if len(tri_k) == 0 or len(tei_k) == 0:
            reason = "attention filter emptied train or test block"
            log(f"[wf] fold {sp['fold']}: SKIPPED ({reason})")
            skipped.append({"fold": sp["fold"], "reason": reason})
            continue
        # Per-fold cutoff from train abnormal returns only.
        cutoff = vlabels.fold_cutoff(work["abn_ret"].iloc[tri_k], label_cfg.quantile)
        yte = vlabels.fold_labels(work["abn_ret"].iloc[tei_k], cutoff).to_numpy()

        base_tri, ytr = _fold_train_labels(work, tri_k, cutoff, train_label_col)
        if dedupe_policy == "first":
            tri_eff = dedupe.dedupe_first(work.iloc[base_tri]).index.to_numpy()
            sample_weight = None
        elif dedupe_policy == "weight":
            tri_eff = base_tri
            sample_weight = dedupe.sample_weights(work.iloc[tri_eff])
        else:
            raise ValueError(f"unknown dedupe_policy {dedupe_policy!r}")
        if train_label_col is not None:
            ytr = work[train_label_col].to_numpy(dtype=float)[tri_eff].astype(int)
        else:
            ytr = vlabels.fold_labels(work["abn_ret"].iloc[tri_eff], cutoff).to_numpy()

        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            reason = (
                f"train classes={sorted(map(int, np.unique(ytr)))}, "
                f"test classes={sorted(map(int, np.unique(yte)))}"
            )
            log(f"[wf] fold {sp['fold']}: SKIPPED ({reason})")
            skipped.append({"fold": sp["fold"], "reason": reason})
            continue
        if len(tri_eff) < min_train:
            reason = f"only {len(tri_eff)} train rows (min_train={min_train})"
            log(f"[wf] fold {sp['fold']}: SKIPPED ({reason})")
            skipped.append({"fold": sp["fold"], "reason": reason})
            continue

        if sp["refit"] or last_fit is None:
            # Vectorizer fit on train texts only: no vocabulary leakage.
            vec = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True,
            )
            vec.fit(texts.iloc[tri_eff])
            Xtr = featurize_fn(
                texts.iloc[tri_eff], vec, dense_extra=_take_rows(dense_extra, tri_eff)
            )
            fitted = fit_challenger_candidates(Xtr, ytr, sample_weight=sample_weight)
            model = _resolve_model(fitted, model_name)
            last_fit = (vec, model)
        else:
            vec, model = last_fit
        Xte = featurize_fn(
            texts.iloc[tei_k], vec, dense_extra=_take_rows(dense_extra, tei_k)
        )
        proba = model.predict_proba(Xte)[:, 1]
        rep = _metrics(model_name, yte, proba)
        base_pr_auc = float(yte.mean())  # majority-baseline PR-AUC = pos rate
        if return_oos:
            for p, s in zip(tei_k.tolist(), proba.tolist()):
                oos[int(p)] = float(s)

        folds.append(
            {
                "fold": sp["fold"],
                "test_start": sp["test_start"],
                "test_end": sp["test_end"],
                "n_train": int(len(tri_eff)),
                "n_test": int(len(tei_k)),
                "n_train_pos": int(ytr.sum()),
                "n_test_pos": int(yte.sum()),
                "n_purged": sp["n_purged"],
                "n_embargoed": sp["n_embargoed"],
                "refit": bool(sp["refit"]),
                "cutoff": round(float(cutoff), 6),
                "pr_auc": round(rep.pr_auc, 4),
                "roc_auc": round(rep.roc_auc, 4),
                "f1": round(rep.f1, 4),
                "precision": round(rep.precision, 4),
                "recall": round(rep.recall, 4),
                "baseline_pr_auc": round(base_pr_auc, 4),
            }
        )
        log(
            f"[wf] fold {sp['fold']}: train={len(tri_eff)} test={len(tei_k)} "
            f"pos_rate={yte.mean():.3f} pr_auc={rep.pr_auc:.4f} "
            f"roc_auc={rep.roc_auc:.4f} f1={rep.f1:.4f}"
            + ("" if sp["refit"] else " (fit reused)")
        )

    def _agg(key):
        vals = np.array([f[key] for f in folds], dtype=float)
        if len(vals) == 0:
            raise ValueError(
                "walk-forward: all folds were skipped (single-class train/test); "
                "the dataset is too small or too imbalanced for "
                f"{n_splits} folds"
            )
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

    result = {
        "n_splits": n_splits,
        "horizon_days": label_cfg.window_days,
        "embargo_days": embargo_days,
        "purge": "exact [t0,t1] overlap (validation.splits)",
        "model": model_name,
        "dedupe_policy": dedupe_policy,
        "validation": {
            "window": window,
            "train_window_days": train_window_days,
            "retrain_every": retrain_every,
            "dev_end": eff_dev_end.isoformat(),
            "periods": periods.describe(),
        },
        "train_labels": {
            "source": train_label_source
            if train_label_source is not None
            else ("weak" if train_label_col is None else train_label_col),
            "test_labels": "weak per-fold (cutoff from train abn_ret only)",
        },
        "folds": folds,
        "skipped_folds": skipped,
        "aggregate": {
            "pr_auc": _agg("pr_auc"),
            "roc_auc": _agg("roc_auc"),
            "f1": _agg("f1"),
            "precision": _agg("precision"),
            "recall": _agg("recall"),
            "baseline_pr_auc": _agg("baseline_pr_auc"),
        },
    }
    if return_oos:
        result["oos_proba"] = oos
    return result


def final_train_and_test(
    df: pd.DataFrame,
    dense_extra=None,
    trading_days: np.ndarray | None = None,
    label_cfg: LabelConfig | None = None,
    min_train: int = 50,
    dedupe_policy: str = "weight",
    model_name: str = "logreg_balanced",
    featurize_fn=None,
    log=print,
    train_label_col: str | None = None,
    train_label_source: str | None = None,
    confirm_test_eval: bool = False,
) -> dict:
    """Final train on all dev (purged vs test) -> single test evaluation.

    Implemented and synthetic-tested in Phase 1; NOT run on real test
    data until the Phase 3 gate. ``confirm_test_eval=True`` is required:
    this touches the untouched test period exactly once per finalized
    configuration. Never used for selection, tuning, thresholds, or
    calibration.

    Steps: dev rows purged against the test range [2026-07-01,
    2026-08-31] (a dev row whose label window reaches into test must not
    train the final model); per-fold-style labels with the cutoff from
    final-train abn_ret only; attention with train-only quantiles;
    dedupe/down-weight; TF-IDF fit on final-train texts only; fit the
    ``model_name`` challenger; score once on the test period.
    """
    if not confirm_test_eval:
        raise RuntimeError(
            "final_train_and_test touches the untouched TEST period; pass "
            "confirm_test_eval=True only at the Phase 3 gate, once per "
            "finalized configuration"
        )
    label_cfg = (label_cfg or LabelConfig()).validated()
    if trading_days is None:
        raise ValueError("trading_days (union calendar ordinals) is required")
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"model_name must be one of {MODEL_CHOICES}, got {model_name!r}")

    work = df.dropna(subset=["published_at", "t0", "t1", "abn_ret"]).reset_index(drop=True)
    if "text" not in work.columns:
        work = work.copy()
        work["text"] = make_text(work)
    point_in_time.check_frame(work, "final_train_and_test")
    periods.check_no_confirmation(work["t0"], "final_train_and_test")

    t0d = pd.to_datetime(work["t0"]).dt.date
    dev_mask = t0d <= periods.DEV_END
    test_mask = (t0d >= periods.TEST_START) & (t0d <= periods.TEST_END)
    dev, test = work[dev_mask], work[test_mask]
    if len(test) == 0:
        raise ValueError("[final] no rows in the test period 2026-07-01..2026-08-31")
    keep = splits.purge_against(
        dev["t0"], dev["t1"], periods.TEST_START, periods.TEST_END
    )
    train_df = dev[keep.values].reset_index(drop=True)
    n_purged = int((~keep).sum())
    log(f"[final] train={len(train_df)} (purged {n_purged} vs test), test={len(test)}")
    if len(train_df) < min_train:
        raise ValueError(
            f"[final] only {len(train_df)} train rows (min_train={min_train})"
        )

    full = pd.concat([train_df, test], ignore_index=True)
    if label_cfg.attention_enabled:
        mask = vlabels.attention_keep_mask(
            train_df,
            full,
            min_articles=label_cfg.attention_min_articles,
            top_quartile=label_cfg.attention_top_quartile,
        ).to_numpy()
    else:
        mask = np.ones(len(full), dtype=bool)
    n_tr = len(train_df)
    train_keep = np.flatnonzero(mask[:n_tr])
    test_keep = np.flatnonzero(mask[n_tr:])
    # Map back to positions in work: dev/test were never reset, so their
    # index labels are work positions; keep is aligned to dev.
    dev_wpos = dev.index.to_numpy()[keep.to_numpy()]
    train_wpos = dev_wpos[train_keep]
    test_wpos = test.index.to_numpy()[test_keep]

    cutoff = vlabels.fold_cutoff(work["abn_ret"].iloc[train_wpos], label_cfg.quantile)
    yte = vlabels.fold_labels(work["abn_ret"].iloc[test_wpos], cutoff).to_numpy()
    base_wpos, ytr0 = _fold_train_labels(work, train_wpos, cutoff, train_label_col)
    if dedupe_policy == "first":
        tri_eff = dedupe.dedupe_first(work.iloc[base_wpos]).index.to_numpy()
        sample_weight = None
    elif dedupe_policy == "weight":
        tri_eff = base_wpos
        sample_weight = dedupe.sample_weights(work.iloc[tri_eff])
    else:
        raise ValueError(f"unknown dedupe_policy {dedupe_policy!r}")
    if train_label_col is not None:
        ytr = work[train_label_col].to_numpy(dtype=float)[tri_eff].astype(int)
    else:
        ytr = vlabels.fold_labels(work["abn_ret"].iloc[tri_eff], cutoff).to_numpy()

    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        raise ValueError(
            "[final] single-class train or test: "
            f"train={sorted(map(int, np.unique(ytr)))}, "
            f"test={sorted(map(int, np.unique(yte)))}"
        )

    texts = work["text"].fillna("").astype(str)
    featurize_fn = featurize_fn or featurize
    vec = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True
    )
    vec.fit(texts.iloc[tri_eff])
    Xtr = featurize_fn(
        texts.iloc[tri_eff], vec, dense_extra=_take_rows(dense_extra, tri_eff)
    )
    fitted = fit_challenger_candidates(Xtr, ytr, sample_weight=sample_weight)
    model = _resolve_model(fitted, model_name)
    Xte = featurize_fn(
        texts.iloc[test_wpos], vec, dense_extra=_take_rows(dense_extra, test_wpos)
    )
    proba = model.predict_proba(Xte)[:, 1]
    rep = _metrics(f"final_{model_name}", yte, proba)
    log(
        f"[final] test n={len(test_wpos)} pos_rate={yte.mean():.3f} "
        f"pr_auc={rep.pr_auc:.4f} roc_auc={rep.roc_auc:.4f}"
    )
    return {
        "model": model_name,
        "n_train": int(len(tri_eff)),
        "n_test": int(len(test_wpos)),
        "n_purged_vs_test": n_purged,
        "test_start": periods.TEST_START.isoformat(),
        "test_end": periods.TEST_END.isoformat(),
        "cutoff": round(float(cutoff), 6),
        "pr_auc": round(rep.pr_auc, 4),
        "roc_auc": round(rep.roc_auc, 4),
        "f1": round(rep.f1, 4),
        "precision": round(rep.precision, 4),
        "recall": round(rep.recall, 4),
        "baseline_pr_auc": round(float(yte.mean()), 4),
        "train_labels": {
            "source": train_label_source
            if train_label_source is not None
            else ("weak" if train_label_col is None else train_label_col),
            "test_labels": "weak (cutoff from final-train abn_ret only)",
        },
    }


def build_provenance(
    representation: str,
    label_cfg,
    df: "pd.DataFrame",
    log=print,
    model_name: str = "logreg_balanced",
) -> dict:
    """Provenance block for a walk-forward run: everything needed to
    interpret the numbers later.

    Records the classifier, representation, label scheme, code revision
    (best-effort git SHA), timestamp, dataset shape/date range, and (for
    FinBERT runs) the embedding-cache coverage. Numbers without this
    block are not comparable across runs.
    """
    import subprocess
    from datetime import datetime, timezone

    try:
        repo_root = Path(__file__).resolve().parents[3]
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        )
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort
        log(f"[wf] provenance: git revision unavailable ({exc})")
        sha, dirty = None, None
    prov = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": sha,
        "working_tree_dirty": dirty,
        "classifier": model_name,
        "representation": representation,
        "label_config": {
            "scheme": label_cfg.scheme,
            "window_days": label_cfg.window_days,
            "quantile": label_cfg.quantile,
            "benchmark": label_cfg.benchmark,
            "attention_enabled": label_cfg.attention_enabled,
            "attention_min_articles": label_cfg.attention_min_articles,
            "attention_top_quartile": label_cfg.attention_top_quartile,
        },
        "validation": {
            "periods": periods.describe(),
            "purge": "exact [t0,t1] overlap (validation.splits)",
        },
        "dataset": {
            "n_rows": int(len(df)),
            "t0_start": str(df["t0"].min()) if "t0" in df else None,
            "t0_end": str(df["t0"].max()) if "t0" in df else None,
        },
    }
    if representation == "finbert_ctx":
        from signal_lab.models.embeddings import (
            cache_key,
            default_cache_dir,
        )

        cdir = default_cache_dir()
        texts = df["text"].fillna("").astype(str) if "text" in df else df["title"].fillna("").astype(str)
        keys = {cache_key(t) for t in texts}
        cached = sum((cdir / f"{k}.npy").exists() for k in keys)
        prov["embedding_cache"] = {
            "cache_dir": str(cdir),
            "unique_texts": len(keys),
            "cached": cached,
            "missing": len(keys) - cached,
            "missing_policy": "raise (MissingEmbeddingError; zero-fill only via explicit on_missing='zero')",
        }
    return prov


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Purged walk-forward CV with exact [t0,t1] decontamination (0.6.0)"
    )
    ap.add_argument("--folds", type=int, default=5, help="number of test folds")
    ap.add_argument(
        "--repr",
        choices=["tfidf", "finbert_ctx"],
        default="finbert_ctx",
        help="text representation: tfidf (TF-IDF+lexicon) or finbert_ctx "
        "(FinBERT+7 ctx, cache-only, raises MissingEmbeddingError on cache "
        "misses unless on_missing='zero' is passed explicitly)",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="print only the JSON result to stdout (logs to stderr)",
    )
    ap.add_argument(
        "--train-labels",
        choices=["weak", "jev", "jev-conf06"],
        default="weak",
        help="training label source: weak (price-derived, default) or Jev-judged "
        "labels (opt-in). Test folds are always scored against weak labels.",
    )
    ap.add_argument(
        "--jev-labels-path",
        default="data/jev_labels_full.json",
        help="path to the Jev labels JSON (required unless --train-labels weak)",
    )
    ap.add_argument(
        "--window",
        choices=["expanding", "rolling"],
        default="expanding",
        help="walk-forward window: expanding (all history) or rolling "
        "(last --train-window-days of history)",
    )
    ap.add_argument(
        "--train-window-days",
        type=int,
        default=365,
        help="rolling window only: calendar days of train history per fold",
    )
    ap.add_argument(
        "--embargo-days",
        type=int,
        default=5,
        help="embargo after each test period, in trading days (>= horizon)",
    )
    ap.add_argument(
        "--retrain-every",
        type=int,
        default=1,
        help="refit the model every k folds (intermediate folds reuse the fit)",
    )
    ap.add_argument(
        "--dev-end",
        default=None,
        help="YYYY-MM-DD: last t0 included (default: frozen 2026-06-30). "
        "Past the frozen end touches the test period: Phase 3 gate only.",
    )
    ap.add_argument(
        "--dedupe",
        choices=["first", "weight"],
        default="weight",
        help="same-ticker same-t0 handling: keep first article or "
        "down-weight by group size (default)",
    )
    ap.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="logreg_balanced",
        help="challenger candidate to evaluate per fold",
    )
    ap.add_argument(
        "--include-test",
        action="store_true",
        help="LOUD: include the untouched test period in the folds. "
        "Phase 3 gate only; never for selection.",
    )
    ap.add_argument(
        "--final",
        action="store_true",
        help="final train on all dev (purged vs test) -> single test "
        "evaluation. Phase 3 gate only; not a development command.",
    )

    # Late import: building the dataset needs the DB and the full pipeline.
    from signal_lab.models.build_and_train import (
        build_dataset,
        load_trading_calendar_days,
    )
    from signal_lab.models.context_features import CONTEXT_FEATURE_NAMES
    from signal_lab.models.run import add_label_args, label_config_from_args
    from signal_lab.models.train_labels import (
        TRAIN_LABEL_COL,
        apply_train_labels,
    )

    add_label_args(ap)
    args = ap.parse_args()
    log = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print

    label_cfg = label_config_from_args(args).validated()
    df, _label_block = build_dataset(log=log, label_cfg=label_cfg)
    if args.train_labels != "weak":
        df = apply_train_labels(
            df, args.train_labels, jev_path=args.jev_labels_path, log=log
        )
        log(f"[wf] train labels: source={args.train_labels}")
    train_label_col = TRAIN_LABEL_COL if args.train_labels != "weak" else None
    trading_days = load_trading_calendar_days(log=log)
    ctx = df[CONTEXT_FEATURE_NAMES]
    featurize_fn = None
    if args.repr == "finbert_ctx":
        from signal_lab.models.embeddings import finbert_ctx_matrix

        def featurize_fn(texts, vectorizer, dense_extra=None):
            return finbert_ctx_matrix(texts, dense_extra=dense_extra, log=log)

    dev_end = date.fromisoformat(args.dev_end) if args.dev_end else None
    if args.final:
        log(
            "[wf] --final: FINAL evaluation on the untouched test period. "
            "Phase 3 gate only; this run must not inform any selection."
        )
        result = final_train_and_test(
            df,
            dense_extra=ctx,
            trading_days=trading_days,
            label_cfg=label_cfg,
            dedupe_policy=args.dedupe,
            model_name=args.model,
            featurize_fn=featurize_fn,
            log=log,
            train_label_col=train_label_col,
            train_label_source=args.train_labels,
            confirm_test_eval=True,
        )
    else:
        result = run_walk_forward(
            df,
            dense_extra=ctx,
            n_splits=args.folds,
            trading_days=trading_days,
            label_cfg=label_cfg,
            window=args.window,
            train_window_days=args.train_window_days,
            embargo_days=args.embargo_days,
            retrain_every=args.retrain_every,
            dev_end=dev_end,
            dedupe_policy=args.dedupe,
            include_test=args.include_test,
            model_name=args.model,
            featurize_fn=featurize_fn,
            log=log,
            train_label_col=train_label_col,
            train_label_source=args.train_labels,
        )
    result["representation"] = args.repr
    result["provenance"] = build_provenance(
        args.repr, label_cfg, df, log=log, model_name=args.model
    )
    print(json.dumps(sanitize_json(result), indent=2))


if __name__ == "__main__":
    main()
