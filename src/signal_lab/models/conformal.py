"""Split-conformal prediction sets with a principled abstention rule.

Replaces the ad-hoc confidence-margin abstention in
``signal_lab.models.calibration`` (gate-1 finding: abstention hurt F1 at
every tested margin). Split conformal prediction carries a
finite-sample marginal coverage guarantee: with nonconformity scores
``s_i = 1 - p_hat(y_i | x_i)`` on a held-out calibration block and the
``ceil((n_cal + 1) * (1 - alpha)) / n_cal`` quantile ``q_hat``, the
prediction set ``C(x) = {y : 1 - p_hat(y | x) <= q_hat}`` contains the
true label with probability at least ``1 - alpha`` under exchangeability
(Vovk et al.; the MAPIE package implements the same construction for
sklearn estimators — this module reimplements the split-conformal core
in our typed style so the walk-forward harness owns the temporal
logic, with no new dependency).

Abstention rule: ABSTAIN when the conformal set is empty or
non-singleton (the model cannot commit to exactly one label at the
target confidence); ACT with the singleton label otherwise.

Time-series drift breaks exchangeability, so calibration is ROLLING:
per fold, the quantile is fit on the most recent ``cal_window`` rows of
the fold's calibration block (temporal order). The quantile is never
reused across folds.

Integration: ``run_conformal_walk_forward`` reuses the validation
harness (``splits.make_splits``, per-fold train-only cutoffs, purge +
embargo) and mirrors ``run_walk_forward``'s fold pipeline, except the
fold train block is split temporally into proper-train (model +
vectorizer fit) and calibration (quantile fit). Test folds are scored
only through prediction sets.

Usage:
    python -m signal_lab.models.conformal [--alpha 0.1]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
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
from signal_lab.models.labels import LabelConfig
from signal_lab.validation import dedupe, labels as vlabels, periods, point_in_time, splits

REPO_ROOT = Path(__file__).resolve().parents[3]
ART = REPO_ROOT / "data" / "artifacts"


def nonconformity_scores(y_cal: np.ndarray, proba_cal: np.ndarray) -> np.ndarray:
    """s_i = 1 - p_hat(y_i | x_i): low when the model is right and sure."""
    y = np.asarray(y_cal, dtype=int).ravel()
    p = np.asarray(proba_cal, dtype=float).ravel()
    if len(y) != len(p):
        raise ValueError(f"y_cal ({len(y)}) and proba_cal ({len(p)}) disagree")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("proba_cal must be finite probabilities in [0, 1]")
    p_true = np.where(y == 1, p, 1.0 - p)
    return 1.0 - p_true


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile of calibration scores.

    Uses the ceil((n + 1)(1 - alpha)) / n order statistic, which is what
    gives the 1 - alpha marginal coverage guarantee (not the naive
    np.quantile, which undercovers on small n).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    s = np.asarray(scores, dtype=float).ravel()
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        raise ValueError("no finite calibration scores")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(s)[k - 1])


def prediction_sets(proba: np.ndarray, q_hat: float) -> list[frozenset[int]]:
    """C(x) = {y in {0, 1} : 1 - p_hat(y | x) <= q_hat} per row."""
    p = np.asarray(proba, dtype=float).ravel()
    out: list[frozenset[int]] = []
    for pi in p:
        # Nonconformity of candidate label y is 1 - p_hat(y | x):
        # y = 1 -> 1 - pi; y = 0 -> 1 - (1 - pi) = pi.
        keep = set()
        if pi <= q_hat + 1e-12:
            keep.add(0)
        if 1.0 - pi <= q_hat + 1e-12:
            keep.add(1)
        out.append(frozenset(keep))
    return out


def abstain_mask(sets: list[frozenset[int]]) -> np.ndarray:
    """True where the set is empty or non-singleton: abstain there."""
    return np.array([len(s) != 1 for s in sets], dtype=bool)


def singleton_predictions(sets: list[frozenset[int]]) -> np.ndarray:
    """The committed label where the set is a singleton, -1 on abstain."""
    return np.array([next(iter(s)) if len(s) == 1 else -1 for s in sets], dtype=int)


def empirical_coverage(y_true: np.ndarray, sets: list[frozenset[int]]) -> float:
    """Fraction of rows whose true label is in the prediction set."""
    y = np.asarray(y_true, dtype=int).ravel()
    if len(y) != len(sets):
        raise ValueError("y_true and sets disagree in length")
    if len(y) == 0:
        raise ValueError("empty input")
    return float(np.mean([yi in s for yi, s in zip(y, sets)]))


def fit_rolling_quantile(
    y_cal: np.ndarray,
    proba_cal: np.ndarray,
    alpha: float,
    cal_window: int,
) -> float:
    """Fit the conformal quantile on the most recent ``cal_window``
    calibration rows (caller must pass time-ordered arrays).

    Rolling instead of full-history because return regimes drift; a
    quantile fit on stale calibration data misstates current
    uncertainty. Raises if fewer than ``cal_window`` rows are available
    rather than silently shrinking the window.
    """
    y = np.asarray(y_cal, dtype=int).ravel()
    p = np.asarray(proba_cal, dtype=float).ravel()
    if len(y) < cal_window:
        raise ValueError(
            f"only {len(y)} calibration rows, need cal_window={cal_window}"
        )
    yw, pw = y[-cal_window:], p[-cal_window:]
    return conformal_quantile(nonconformity_scores(yw, pw), alpha)


def run_conformal_walk_forward(
    df: pd.DataFrame,
    dense_extra=None,
    n_splits: int = 5,
    trading_days: np.ndarray | None = None,
    label_cfg: LabelConfig | None = None,
    window: str = "expanding",
    train_window_days: int = 365,
    embargo_days: int = 5,
    alpha: float = 0.1,
    cal_frac: float = 0.2,
    cal_window: int = 100,
    min_train: int = 50,
    dev_end: date | None = None,
    dedupe_policy: str = "weight",
    include_test: bool = False,
    log=print,
) -> dict:
    """Purged walk-forward with split-conformal abstention per fold.

    Same validation discipline as ``run_walk_forward`` (same splits,
    purge, embargo, per-fold train-only cutoffs, dedupe, dev/test
    period guards). Per fold, the train block is split TEMPORALLY into
    proper-train (vectorizer + classifier fit) and calibration (the most
    recent ``cal_frac`` of train rows; the quantile uses the last
    ``cal_window`` of those). The test block is scored only through
    prediction sets: abstain on empty/non-singleton, act on singletons.

    Reports per fold: calibration size, quantile, test-set empirical
    coverage (diagnostic: should sit near 1 - alpha), abstention rate,
    and acted-subset precision/recall/F1. ``include_test`` carries the
    same loud guard as the main harness.
    """
    label_cfg = (label_cfg or LabelConfig()).validated()
    if trading_days is None:
        raise ValueError("trading_days (union calendar ordinals) is required")
    if not 0.0 < cal_frac < 0.5:
        raise ValueError(f"cal_frac must be in (0, 0.5), got {cal_frac}")

    work = df.dropna(subset=["published_at", "t0", "t1", "abn_ret"]).reset_index(drop=True)
    if "text" not in work.columns:
        work = work.copy()
        work["text"] = make_text(work)
    point_in_time.check_frame(work, "run_conformal_walk_forward")

    eff_dev_end = periods.TEST_END if include_test else (dev_end or periods.DEV_END)
    if eff_dev_end > periods.DEV_END:
        log(
            "[cf] WARNING: dev_end "
            f"{eff_dev_end.isoformat()} is past the frozen development end "
            f"{periods.DEV_END.isoformat()}: the untouched TEST period is in "
            "the folds. Never for selection."
        )
    n_before = len(work)
    work = work[pd.to_datetime(work["t0"]).dt.date <= eff_dev_end].reset_index(drop=True)
    log(f"[cf] {n_before - len(work)} rows past {eff_dev_end.isoformat()} excluded")
    config = splits.WalkForwardConfig(
        n_splits=n_splits,
        window=window,
        train_window_days=train_window_days,
        horizon_days=label_cfg.window_days,
        embargo_days=embargo_days,
        purge=True,
        min_train=min_train,
        retrain_every=1,
        dev_end=eff_dev_end,
    )
    wf_splits = splits.make_splits(work, config, np.asarray(trading_days))
    log(
        f"[cf] {n_splits} purged folds ({window}, horizon={label_cfg.window_days}d, "
        f"embargo={embargo_days}d, alpha={alpha}, cal_frac={cal_frac}, "
        f"cal_window={cal_window})"
    )

    texts = work["text"].fillna("").astype(str)
    folds, skipped = [], []
    for sp in wf_splits:
        tri, tei = sp["train_idx"], sp["test_idx"]
        # Temporal order inside the train block: calibration is the most
        # recent cal_frac of train rows (rolling window needs recency).
        t_tri = pd.to_datetime(work["published_at"].iloc[tri], utc=True).values
        order = np.argsort(t_tri, kind="stable")
        tri_sorted = tri[order]
        n_cal = max(1, int(len(tri_sorted) * cal_frac))
        proper_idx, cal_idx = tri_sorted[:-n_cal], tri_sorted[-n_cal:]

        cutoff = vlabels.fold_cutoff(work["abn_ret"].iloc[tri], label_cfg.quantile)
        yte = vlabels.fold_labels(work["abn_ret"].iloc[tei], cutoff).to_numpy()
        y_proper = vlabels.fold_labels(work["abn_ret"].iloc[proper_idx], cutoff).to_numpy()
        y_cal = vlabels.fold_labels(work["abn_ret"].iloc[cal_idx], cutoff).to_numpy()

        if (
            len(np.unique(y_proper)) < 2
            or len(np.unique(y_cal)) < 2
            or len(np.unique(yte)) < 2
        ):
            reason = "single-class proper-train, calibration, or test block"
            log(f"[cf] fold {sp['fold']}: SKIPPED ({reason})")
            skipped.append({"fold": sp["fold"], "reason": reason})
            continue
        if len(proper_idx) < min_train or len(cal_idx) < cal_window:
            reason = (
                f"proper={len(proper_idx)} (min {min_train}) or "
                f"calibration={len(cal_idx)} (window {cal_window}) too small"
            )
            log(f"[cf] fold {sp['fold']}: SKIPPED ({reason})")
            skipped.append({"fold": sp['fold'], "reason": reason})
            continue

        if dedupe_policy == "weight":
            sw = dedupe.sample_weights(work.iloc[proper_idx])
        elif dedupe_policy == "first":
            keep_pos = dedupe.dedupe_first(work.iloc[proper_idx]).index.to_numpy()
            proper_idx, y_proper = keep_pos, vlabels.fold_labels(
                work["abn_ret"].iloc[keep_pos], cutoff
            ).to_numpy()
            sw = None
        else:
            raise ValueError(f"unknown dedupe_policy {dedupe_policy!r}")

        vec = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            stop_words="english", sublinear_tf=True,
        )
        vec.fit(texts.iloc[proper_idx])
        Xpr = featurize(texts.iloc[proper_idx], vec,
                        dense_extra=_take_rows(dense_extra, proper_idx))
        clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                 random_state=RANDOM_STATE)
        clf.fit(Xpr, y_proper, sample_weight=sw)
        Xca = featurize(texts.iloc[cal_idx], vec,
                        dense_extra=_take_rows(dense_extra, cal_idx))
        q_hat = fit_rolling_quantile(y_cal, clf.predict_proba(Xca)[:, 1],
                                     alpha, cal_window)

        Xte = featurize(texts.iloc[tei], vec,
                        dense_extra=_take_rows(dense_extra, tei))
        pte = clf.predict_proba(Xte)[:, 1]
        sets = prediction_sets(pte, q_hat)
        abst = abstain_mask(sets)
        acted = singleton_predictions(sets)

        coverage = empirical_coverage(yte, sets)
        abst_rate = float(abst.mean())
        acted_row: dict = {
            "fold": sp["fold"],
            "test_start": sp["test_start"],
            "test_end": sp["test_end"],
            "n_proper": int(len(proper_idx)),
            "n_cal": int(len(cal_idx)),
            "q_hat": round(q_hat, 4),
            "n_test": int(len(tei)),
            "empirical_coverage": round(coverage, 4),
            "target_coverage": round(1.0 - alpha, 4),
            "abstention_rate": round(abst_rate, 4),
            "n_acted": int((~abst).sum()),
        }
        if (~abst).sum() > 0 and len(np.unique(yte[~abst])) == 2:
            rep = _metrics("acted", yte[~abst], pte[~abst])
            acted_row.update(
                {
                    "acted_precision": round(rep.precision, 4),
                    "acted_recall": round(rep.recall, 4),
                    "acted_f1": round(rep.f1, 4),
                }
            )
        else:
            acted_row.update(
                {"acted_precision": None, "acted_recall": None,
                 "acted_f1": None,
                 "note": "acted subset empty or single-class"}
            )
        folds.append(acted_row)
        log(
            f"[cf] fold {sp['fold']}: n_cal={len(cal_idx)} q={q_hat:.3f} "
            f"coverage={coverage:.3f} (target {1 - alpha:.2f}) "
            f"abstain={abst_rate:.3f} acted_f1={acted_row['acted_f1']}"
        )

    return {
        "method": "split-conformal abstention (rolling calibration window)",
        "alpha": alpha,
        "cal_frac": cal_frac,
        "cal_window": cal_window,
        "n_splits": n_splits,
        "dev_end": eff_dev_end.isoformat(),
        "folds": folds,
        "skipped_folds": skipped,
        "aggregate": {
            "mean_coverage": round(float(np.mean([f["empirical_coverage"] for f in folds])), 4)
            if folds else None,
            "mean_abstention_rate": round(float(np.mean([f["abstention_rate"] for f in folds])), 4)
            if folds else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split-conformal abstention on the purged walk-forward"
    )
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="miscoverage level (target coverage 1 - alpha)")
    ap.add_argument("--json", action="store_true",
                    help="print only the JSON result to stdout (logs to stderr)")
    args = ap.parse_args()
    log = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print

    from signal_lab.models.build_and_train import (
        build_dataset,
        load_trading_calendar_days,
    )
    from signal_lab.models.context_features import CONTEXT_FEATURE_NAMES
    from signal_lab.models.run import add_label_args, label_config_from_args

    add_label_args(ap)
    args = ap.parse_args()
    log = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print

    label_cfg = label_config_from_args(args).validated()
    df, _ = build_dataset(log=log, label_cfg=label_cfg)
    trading_days = load_trading_calendar_days(log=log)
    result = run_conformal_walk_forward(
        df,
        dense_extra=df[CONTEXT_FEATURE_NAMES],
        trading_days=trading_days,
        label_cfg=label_cfg,
        alpha=args.alpha,
        log=log,
    )
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "conformal.json").write_text(json.dumps(sanitize_json(result), indent=2))
    log(f"[cf] wrote {ART / 'conformal.json'}")
    if args.json:
        print(json.dumps(sanitize_json(result)))


if __name__ == "__main__":
    main()
