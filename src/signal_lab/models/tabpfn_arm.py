"""TabPFN zero-tuning benchmark arm for the small-n regime.

TabPFN is a tabular foundation model: it makes predictions by
in-context conditioning on the training set, with no hyperparameter
tuning. That makes it the right "what does zero tuning buy?" benchmark
against our tuned logreg / LightGBM challengers on small samples.

Version pin (deliberate): ``tabpfn==2.0.6`` is the last release that
runs fully locally with no account. tabpfn>=2.5 requires a PriorLabs
account and a TABPFN_TOKEN (browser login), which is incompatible with
headless CI and our no-credentials rule. The 2.0.6 ``huggingface-hub<1``
pin is stale (it runs fine on hub>=1.31, verified); CI installs it with
``--no-deps`` so the pin cannot downgrade the hub under transformers.
Model weights (~100MB) download from HuggingFace on first fit and are
cached in the platform cache dir.

Point-in-time discipline: TabPFN's "training" IS its conditioning set.
Per fold we condition ONLY on that fold's train block (rows strictly
before the fold's test range, purged + embargoed by the validation
harness). No future fold ever enters a conditioning set. PCA (needed
because TF-IDF's 5000 features exceed TabPFN-v2's feature budget) is
fit per fold on the train block only.

Dedupe: TabPFNClassifier takes no sample weights, so folds use
dedupe-first (keep earliest article per ticker-t0) rather than the
weight policy; the choice is explicit, never silent.

Usage:
    python -m signal_lab.models.tabpfn_arm [--folds 5] [--json]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

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

PCA_COMPONENTS = 50  # comfortably inside TabPFN-v2's feature budget
TABPFN_PIN = "tabpfn==2.0.6"


def _require_tabpfn():
    if importlib.util.find_spec("tabpfn") is None:
        raise RuntimeError(
            f"[tabpfn] the tabpfn package is not installed; install {TABPFN_PIN} "
            "(last fully-local release; >=2.5 needs a PriorLabs account token). "
            "See the module docstring for the huggingface-hub pin workaround."
        )
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier


def fit_predict_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Zero-tuning TabPFN: condition on (X_train, y_train), return P(y=1).

    No hyperparameters are set beyond the device; the model is used
    exactly as shipped. Raises if the training block is single-class
    (TabPFN needs at least two classes to condition on).
    """
    TabPFNClassifier = _require_tabpfn()
    Xtr = np.asarray(X_train, dtype=np.float32)
    Xte = np.asarray(X_test, dtype=np.float32)
    ytr = np.asarray(y_train, dtype=int).ravel()
    if len(np.unique(ytr)) < 2:
        raise ValueError("[tabpfn] single-class conditioning set; refusing to fit")
    if Xtr.shape[1] > 100:
        raise ValueError(
            f"[tabpfn] {Xtr.shape[1]} features exceeds TabPFN-v2's budget; "
            "reduce PCA_COMPONENTS"
        )
    clf = TabPFNClassifier(device=device)
    clf.fit(Xtr, ytr)
    return np.asarray(clf.predict_proba(Xte)[:, 1], dtype=float)


def pca_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = PCA_COMPONENTS,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """PCA fit on TRAIN only, applied to test. Returns (Z_train, Z_test, pca).

    Constant train columns are dropped first: they carry no signal and
    make TabPFN's internal encoder divide by zero (NaN). This is a
    no-information filter, not scaling, so the zero-tuning spirit holds.
    """
    Xtr = np.asarray(X_train, dtype=float)
    Xte = np.asarray(X_test, dtype=float)
    keep = Xtr.std(axis=0) > 1e-12
    n_dropped = int((~keep).sum())
    if n_dropped:
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    n_comp = min(n_components, Xtr.shape[0] - 1, Xtr.shape[1])
    if n_comp < 2:
        raise ValueError(
            f"[tabpfn] cannot fit PCA: train has {Xtr.shape[0]} rows, "
            f"{Xtr.shape[1]} non-constant features"
        )
    pca = PCA(n_components=n_comp, random_state=random_state)
    Ztr = pca.fit_transform(Xtr)
    Zte = pca.transform(Xte)
    if not (np.all(np.isfinite(Ztr)) and np.all(np.isfinite(Zte))):
        raise ValueError(
            "[tabpfn] non-finite values in PCA output; refusing to condition"
        )
    return Ztr, Zte, pca


def run_tabpfn_walk_forward(
    df: pd.DataFrame,
    dense_extra=None,
    n_splits: int = 5,
    trading_days: np.ndarray | None = None,
    label_cfg: LabelConfig | None = None,
    window: str = "expanding",
    train_window_days: int = 365,
    embargo_days: int = 5,
    min_train: int = 50,
    dev_end: date | None = None,
    include_test: bool = False,
    device: str = "cpu",
    log=print,
) -> dict:
    """Purged walk-forward with the TabPFN arm.

    Same validation discipline as ``run_walk_forward`` (same splits,
    purge, embargo, per-fold train-only cutoffs, dev/test period
    guards), except: dedupe-first (TabPFN takes no sample weights),
    TF-IDF fit per fold on train texts only, PCA fit per fold on the
    train matrix only, then TabPFN conditions on the fold's train block
    alone. Test folds are always scored against per-fold weak labels,
    so the evaluation target matches the logreg/LightGBM arms exactly.
    """
    _require_tabpfn()  # fail loud before any work if tabpfn is missing
    label_cfg = (label_cfg or LabelConfig()).validated()
    if trading_days is None:
        raise ValueError("trading_days (union calendar ordinals) is required")

    work = df.dropna(subset=["published_at", "t0", "t1", "abn_ret"]).reset_index(drop=True)
    if "text" not in work.columns:
        work = work.copy()
        work["text"] = make_text(work)
    point_in_time.check_frame(work, "run_tabpfn_walk_forward")

    eff_dev_end = periods.TEST_END if include_test else (dev_end or periods.DEV_END)
    if eff_dev_end > periods.DEV_END:
        log(
            "[tabpfn] WARNING: dev_end "
            f"{eff_dev_end.isoformat()} is past the frozen development end "
            f"{periods.DEV_END.isoformat()}: the untouched TEST period is in "
            "the folds. Never for selection."
        )
    n_before = len(work)
    work = work[pd.to_datetime(work["t0"]).dt.date <= eff_dev_end].reset_index(drop=True)
    log(f"[tabpfn] {n_before - len(work)} rows past {eff_dev_end.isoformat()} excluded")
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
    log(f"[tabpfn] {n_splits} purged folds ({window}, TabPFN zero-tuning arm)")

    texts = work["text"].fillna("").astype(str)
    folds, skipped = [], []
    for sp in wf_splits:
        tri, tei = sp["train_idx"], sp["test_idx"]
        # Point-in-time assertion: every conditioning row strictly
        # predates the test block. (The harness guarantees this; the
        # assert makes the arm's contract explicit and loud.)
        t0_tri = pd.to_datetime(work["t0"].iloc[tri])
        assert bool((t0_tri < sp["test_start"]).all()), (
            f"[tabpfn] fold {sp['fold']}: conditioning set reaches into the test range"
        )
        cutoff = vlabels.fold_cutoff(work["abn_ret"].iloc[tri], label_cfg.quantile)
        yte = vlabels.fold_labels(work["abn_ret"].iloc[tei], cutoff).to_numpy()

        # Dedupe-first: TabPFN takes no sample weights, so down-weighting
        # is not an option; keep the earliest article per ticker-t0.
        # (work's index is positional after reset_index, so these are
        # direct positions into work.)
        tri_eff = dedupe.dedupe_first(work.iloc[tri]).index.to_numpy()
        ytr = vlabels.fold_labels(work["abn_ret"].iloc[tri_eff], cutoff).to_numpy()

        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            reason = "single-class train or test block"
            log(f"[tabpfn] fold {sp['fold']}: SKIPPED ({reason})")
            skipped.append({"fold": sp["fold"], "reason": reason})
            continue
        if len(tri_eff) < min_train:
            reason = f"only {len(tri_eff)} train rows (min_train={min_train})"
            log(f"[tabpfn] fold {sp['fold']}: SKIPPED ({reason})")
            skipped.append({"fold": sp["fold"], "reason": reason})
            continue

        vec = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            stop_words="english", sublinear_tf=True,
        )
        vec.fit(texts.iloc[tri_eff])
        Xtr_full = featurize(texts.iloc[tri_eff], vec,
                             dense_extra=_take_rows(dense_extra, tri_eff))
        Xte_full = featurize(texts.iloc[tei], vec,
                             dense_extra=_take_rows(dense_extra, tei))
        Xtr = np.asarray(Xtr_full.todense() if hasattr(Xtr_full, "todense") else Xtr_full,
                         dtype=float)
        Xte = np.asarray(Xte_full.todense() if hasattr(Xte_full, "todense") else Xte_full,
                         dtype=float)
        Ztr, Zte, pca = pca_features(Xtr, Xte)
        proba = fit_predict_tabpfn(Ztr, ytr, Zte, device=device)

        rep = _metrics("tabpfn", yte, proba)
        base_pr_auc = float(yte.mean())
        folds.append(
            {
                "fold": sp["fold"],
                "test_start": sp["test_start"],
                "test_end": sp["test_end"],
                "n_train": int(len(tri_eff)),
                "n_test": int(len(tei)),
                "n_test_pos": int(yte.sum()),
                "n_purged": sp["n_purged"],
                "n_embargoed": sp["n_embargoed"],
                "pca_components": int(pca.n_components_),
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
            f"[tabpfn] fold {sp['fold']}: train={len(tri_eff)} test={len(tei)} "
            f"pos_rate={yte.mean():.3f} pr_auc={rep.pr_auc:.4f} "
            f"roc_auc={rep.roc_auc:.4f} f1={rep.f1:.4f}"
        )

    def _agg(key):
        vals = np.array([f[key] for f in folds], dtype=float)
        if len(vals) == 0:
            raise ValueError("[tabpfn] all folds were skipped")
        return {"mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0}

    return {
        "model": "tabpfn_zero_tuning",
        "tabpfn_pin": TABPFN_PIN,
        "n_splits": n_splits,
        "validation": {
            "window": window,
            "dev_end": eff_dev_end.isoformat(),
            "dedupe": "first (TabPFN takes no sample weights)",
            "pca": f"{PCA_COMPONENTS} components, fit per fold on train only",
            "conditioning": "per-fold train block only (point-in-time, asserted)",
        },
        "folds": folds,
        "skipped_folds": skipped,
        "aggregate": {
            "pr_auc": _agg("pr_auc"),
            "roc_auc": _agg("roc_auc"),
            "f1": _agg("f1"),
            "baseline_pr_auc": _agg("baseline_pr_auc"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="TabPFN zero-tuning walk-forward arm")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--json", action="store_true",
                    help="print only the JSON result to stdout (logs to stderr)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="TabPFN device")
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
    result = run_tabpfn_walk_forward(
        df,
        dense_extra=df[CONTEXT_FEATURE_NAMES],
        n_splits=args.folds,
        trading_days=trading_days,
        label_cfg=label_cfg,
        device=args.device,
        log=log,
    )
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "tabpfn_arm.json").write_text(json.dumps(sanitize_json(result), indent=2))
    log(f"[tabpfn] wrote {ART / 'tabpfn_arm.json'}")
    if args.json:
        print(json.dumps(sanitize_json(result)))


if __name__ == "__main__":
    main()
