"""Phase 3: benchmark and go/no-go gate (research program).

PRE-SPECIFIED GATE RULE (frozen before the run; see docs/GATE.md):

  primary metric    : mean PR-AUC across the dev walk-forward folds
  primary model     : logreg_balanced
  primary comparison: variant C (text + price) minus variant A (price-only),
                      paired per fold
  GO  iff the mean paired difference is > 0 AND the lower bound of the
      two-sided 95% Student-t confidence interval (4 df) is > 0.
  Otherwise STOP: no test-period evaluation, no phases 4-7.

Design notes:

- Everything here runs on the DEVELOPMENT period only (t0 <= 2026-06-30).
  The test period (2026-07-01..2026-08-31) is touched exactly once, and
  only after a GO verdict, via
  ``signal_lab.models.walk_forward.final_train_and_test``
  (``confirm_test_eval=True``). The confirmation period is never touched.
- Splits, purge, embargo, per-fold attention (train-only quantiles),
  per-fold cutoff (train-only), dedupe weights, and TF-IDF (fit on train
  texts only) all reuse the ``signal_lab.validation`` primitives; a test
  pins the fold test indices identical to ``run_walk_forward``'s.
- Variants: A = price-only (scaled dense price/volume/market features),
  B = text-only (TF-IDF + lexicon), C = combined. The median imputer and
  the standard scaler are fit on train data only, per fold.
- Models: naive (constant 0.5), historical_rate (constant train positive
  rate), logreg_plain, logreg_balanced, lightgbm_balanced.
- Calibration (Platt / isotonic via CalibratedClassifierCV with a
  TimeSeriesSplit(3) inside the train block) is a diagnostic on
  logreg_balanced: Brier/ECE before vs after. It cannot move the gate:
  PR-AUC is rank-based and monotone recalibration preserves rankings.
- Abstention (|p - 0.5| < margin) is evaluated on the primary
  configuration (variant C, logreg_balanced, raw probas): coverage and
  retained-set metrics per fold. It is kept only if measured to help
  (retained F1 > full F1); the verdict is reported in docs/GATE.md.
- Deterministic: all stochastic estimators take BenchmarkConfig.seed;
  the only randomness in the pipeline is inside those estimators.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from signal_lab import sanitize_json
from signal_lab.models import RANDOM_STATE, make_text
from signal_lab.models.context_features import CONTEXT_FEATURE_NAMES, PRICE_FEATURE_NAMES
from signal_lab.models.labels import LabelConfig
from signal_lab.models.lexicon import lexicon_feature_matrix
from signal_lab.validation import dedupe, labels as vlabels, periods, point_in_time, splits
from signal_lab.validation.targets import TARGET_SET_VERSION

try:
    import lightgbm as lgb

    _LGBM_OK = True
except Exception:  # noqa: BLE001 - LightGBM optional; absence is loud, not silent
    _LGBM_OK = False

REPO_ROOT = Path(__file__).resolve().parents[3]
ART = REPO_ROOT / "data" / "artifacts"

VARIANTS = ("A", "B", "C")
VARIANT_DESCR = {
    "A": "price-only: scaled dense price/volume/market features",
    "B": "text-only: TF-IDF (5000, 1-2gram, sublinear) + 8 lexicon features",
    "C": "combined: TF-IDF + lexicon + scaled dense price features",
}
MODEL_NAMES = (
    "naive",
    "historical_rate",
    "logreg_plain",
    "logreg_balanced",
    "lightgbm_balanced",
)
PRIMARY_MODEL = "logreg_balanced"
PRIMARY_METRIC = "pr_auc"
ABSTENTION_MARGINS = (0.05, 0.10, 0.15)
CALIB_METHODS = ("sigmoid", "isotonic")

PRICE_COLS = list(CONTEXT_FEATURE_NAMES) + list(PRICE_FEATURE_NAMES)


@dataclass(frozen=True)
class BenchmarkConfig:
    n_splits: int = 5
    window: str = "expanding"
    embargo_days: int = 5
    min_train: int = 50
    dedupe_policy: str = "weight"  # or "first"
    seed: int = RANDOM_STATE
    model_names: tuple[str, ...] = MODEL_NAMES
    calibrate: bool = True
    abstain: bool = True
    min_calib_train: int = 40  # skip calibration below this train size

    def validated(self) -> BenchmarkConfig:
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}")
        if self.window not in ("expanding", "rolling"):
            raise ValueError(f"window must be expanding|rolling, got {self.window!r}")
        if self.embargo_days < 1:
            raise ValueError("embargo_days must be >= 1")
        if self.min_train < 1:
            raise ValueError("min_train must be >= 1")
        if self.dedupe_policy not in ("weight", "first"):
            raise ValueError(f"unknown dedupe_policy {self.dedupe_policy!r}")
        for m in self.model_names:
            if m not in MODEL_NAMES:
                raise ValueError(f"unknown model {m!r}; expected one of {MODEL_NAMES}")
        if self.min_calib_train < 1:
            raise ValueError("min_calib_train must be >= 1")
        return self


def _ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: mean |P - y| per equal-width bin,
    weighted by bin mass."""
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (proba >= lo) & (proba < hi if b < n_bins - 1 else proba <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(float(proba[m].mean()) - float(y_true[m].mean()))
    return float(ece)


def fold_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    excess: np.ndarray,
    direction: np.ndarray,
) -> dict:
    """All Phase 3 metrics for one fold's test block. NaN-safe: constant
    probas (the naive/historical_rate baselines) yield pr_auc = pos_rate,
    roc_auc = 0.5, rank_ic = NaN instead of raising."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    out: dict[str, float] = {}
    out["pr_auc"] = float(average_precision_score(y_true, proba))
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    except Exception:  # noqa: BLE001 - single-class guard; recorded as NaN
        out["roc_auc"] = float("nan")
    out["brier"] = float(brier_score_loss(y_true, proba))
    out["ece"] = _ece(y_true, proba)
    y_pred = (proba >= 0.5).astype(int)
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    with np.errstate(all="ignore"):
        try:
            out["rank_ic"] = float(
                stats.spearmanr(proba, np.asarray(excess, dtype=float)).statistic
            )
        except Exception:  # noqa: BLE001 - constant input; recorded as NaN
            out["rank_ic"] = float("nan")
    out["directional_accuracy"] = float(
        np.mean(y_pred == np.asarray(direction, dtype=float))
    )
    out["baseline_pr_auc"] = float(y_true.mean())
    return out


def _fit_variant_models(
    mats: dict[str, csr_matrix],
    ytr: np.ndarray,
    sample_weight: np.ndarray | None,
    seed: int,
    model_names: tuple[str, ...],
    log=print,
) -> dict[str, dict[str, object]]:
    """Fit every requested model on every variant matrix. Returns
    variant -> model -> proba-function ``(Xte_dict) -> np.ndarray``.

    Constant baselines need no fit; their proba functions ignore X.
    """
    if sample_weight is not None:
        pos_rate = float(np.average(ytr, weights=sample_weight))
    else:
        pos_rate = float(ytr.mean())
    fitted: dict[str, dict[str, object]] = {v: {} for v in VARIANTS}

    def _const_fn(value: float):
        return lambda Xd: np.full(Xd["A"].shape[0], value)

    for variant in VARIANTS:
        Xtr = mats[variant]
        for name in model_names:
            if name == "naive":
                fitted[variant][name] = _const_fn(0.5)
                continue
            if name == "historical_rate":
                fitted[variant][name] = _const_fn(pos_rate)
                continue
            if name == "logreg_plain":
                est = LogisticRegression(max_iter=1000, random_state=seed)
            elif name == "logreg_balanced":
                est = LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=seed
                )
            elif name == "lightgbm_balanced":
                if not _LGBM_OK:
                    log("[phase3] WARNING: lightgbm not importable; skipping")
                    continue
                neg, pos = int((ytr == 0).sum()), int((ytr == 1).sum())
                est = lgb.LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    scale_pos_weight=neg / max(pos, 1),
                    random_state=seed,
                    verbose=-1,
                )
            else:  # pragma: no cover - validated upstream
                raise ValueError(f"unknown model {name!r}")
            est.fit(Xtr, ytr, sample_weight=sample_weight)
            fitted[variant][name] = (
                lambda Xd, _est=est, _v=variant: np.asarray(
                    _est.predict_proba(Xd[_v])[:, 1]
                )
            )
    return fitted


def _build_matrices(
    texts_tr: pd.Series,
    texts_te: pd.Series,
    dense_tr: pd.DataFrame,
    dense_te: pd.DataFrame,
) -> tuple[dict[str, csr_matrix], dict[str, csr_matrix], TfidfVectorizer]:
    """Variant feature matrices. TF-IDF is fit on train texts only; the
    median imputer and standard scaler are fit on train dense rows only."""
    vec = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True
    )
    vec.fit(list(texts_tr))
    tf_tr = vec.transform(list(texts_tr))
    tf_te = vec.transform(list(texts_te))
    lex_tr = csr_matrix(np.asarray(lexicon_feature_matrix(list(texts_tr)), dtype=float))
    lex_te = csr_matrix(np.asarray(lexicon_feature_matrix(list(texts_te)), dtype=float))

    imp = SimpleImputer(strategy="median")
    dtr = imp.fit_transform(dense_tr.to_numpy(dtype=float))
    dte = imp.transform(dense_te.to_numpy(dtype=float))
    scl = StandardScaler()
    dtr_s = scl.fit_transform(dtr)
    dte_s = scl.transform(dte)
    if not (np.isfinite(dtr_s).all() and np.isfinite(dte_s).all()):
        raise ValueError("[phase3] non-finite values after impute+scale")
    s_tr, s_te = csr_matrix(dtr_s), csr_matrix(dte_s)

    mats_tr = {
        "A": s_tr,
        "B": hstack([tf_tr, lex_tr], format="csr"),
        "C": hstack([tf_tr, lex_tr, s_tr], format="csr"),
    }
    mats_te = {
        "A": s_te,
        "B": hstack([tf_te, lex_te], format="csr"),
        "C": hstack([tf_te, lex_te, s_te], format="csr"),
    }
    return mats_tr, mats_te, vec

def _agg(vals: list[float]) -> dict:
    """Mean/std/t-CI across folds (NaN-aware), same convention as the
    walk_forward module."""
    a = np.asarray(vals, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    out = {"n": n, "mean": float(a.mean()) if n else float("nan")}
    if n > 1:
        sd = float(a.std(ddof=1))
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        out.update(
            {
                "std": sd,
                "ci95": [out["mean"] - half, out["mean"] + half],
                "min": float(a.min()),
                "max": float(a.max()),
            }
        )
    elif n == 1:
        out.update({"std": 0.0, "ci95": [out["mean"], out["mean"]]})
    return out


def _paired_test(a: list[float], b: list[float]) -> dict:
    """Paired comparison a - b across folds: mean diff, t-CI, paired t and
    Wilcoxon. NaN-aware; needs >= 2 non-NaN paired folds."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    m = ~np.isnan(aa) & ~np.isnan(bb)
    d = (aa - bb)[m]
    n = len(d)
    out: dict = {"n": n, "mean_diff": float(d.mean()) if n else float("nan")}
    if n > 1:
        sd = float(d.std(ddof=1))
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        out["ci95"] = [out["mean_diff"] - half, out["mean_diff"] + half]
        t_stat, t_p = stats.ttest_rel(aa[m], bb[m])
        out["t_stat"] = float(t_stat)
        out["t_pvalue"] = float(t_p)
        try:
            w_stat, w_p = stats.wilcoxon(aa[m], bb[m])
            out["wilcoxon_stat"] = float(w_stat)
            out["wilcoxon_pvalue"] = float(w_p)
        except Exception:  # noqa: BLE001 - zero-variance diffs; recorded as NaN
            out["wilcoxon_stat"] = float("nan")
            out["wilcoxon_pvalue"] = float("nan")
    elif n == 1:
        out["ci95"] = [out["mean_diff"], out["mean_diff"]]
    return out


def _calibration_diagnostic(
    mats_tr: dict[str, csr_matrix],
    mats_te: dict[str, csr_matrix],
    ytr: np.ndarray,
    yte: np.ndarray,
    sample_weight: np.ndarray | None,
    seed: int,
    min_calib_train: int,
    log=print,
) -> dict:
    """Platt (sigmoid) and isotonic recalibration of logreg_balanced on
    every variant, fit inside the train block only (TimeSeriesSplit(3)).
    Reports Brier/ECE before vs after. Rank-based PR-AUC is invariant to
    monotone recalibration, so this is a diagnostic, not a gate input.

    A fold is skipped (loudly) when the train block is too small or when
    any internal TimeSeriesSplit train partition is single-class, which
    would make the base estimator unidentifiable. Skips are recorded, not
    hidden."""
    out: dict = {}
    if len(ytr) < min_calib_train:
        log(
            f"[phase3] calibration skipped: train n={len(ytr)} "
            f"< min_calib_train={min_calib_train}"
        )
        return {"skipped": f"train_n={len(ytr)}"}
    cv = TimeSeriesSplit(n_splits=3)
    for _tr, _ in cv.split(ytr):
        if len(np.unique(ytr[_tr])) < 2:
            log(
                "[phase3] calibration skipped: a TimeSeriesSplit(3) internal "
                f"train partition is single-class (train n={len(ytr)})"
            )
            return {"skipped": "single_class_cv_partition"}
    for variant in VARIANTS:
        base = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        )
        base.fit(mats_tr[variant], ytr, sample_weight=sample_weight)
        p_raw = np.asarray(base.predict_proba(mats_te[variant])[:, 1])
        row = {"raw": {"brier": float(brier_score_loss(yte, p_raw)),
                       "ece": _ece(yte, p_raw),
                       "proba_std": float(p_raw.std())}}
        for method in CALIB_METHODS:
            cal = CalibratedClassifierCV(
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=seed
                ),
                method=method,
                cv=TimeSeriesSplit(n_splits=3),
            )
            try:
                cal.fit(mats_tr[variant], ytr, sample_weight=sample_weight)
            except ValueError as e:
                # Degenerate calibration partition (e.g. single-class
                # calibrator fold). Diagnostic-only: record and continue.
                log(f"[phase3] calibration {variant}/{method} skipped: {e}")
                row[method] = {"skipped": str(e)}
                continue
            p_cal = np.asarray(cal.predict_proba(mats_te[variant])[:, 1])
            row[method] = {
                "brier": float(brier_score_loss(yte, p_cal)),
                "ece": _ece(yte, p_cal),
                # A ~zero std means the calibrator collapsed to a constant
                # predictor (common when calibration partitions are tiny);
                # any Brier "improvement" from such a row is degenerate.
                "proba_std": float(p_cal.std()),
            }
        out[variant] = row
    return out


def _abstention_eval(
    proba: np.ndarray,
    yte: np.ndarray,
    excess: np.ndarray,
    direction: np.ndarray,
    margins: tuple[float, ...],
) -> dict:
    """Abstain when |p - 0.5| < margin: coverage and retained-set metrics.
    Compares retained F1 against the no-abstention baseline."""
    out: dict = {}
    full = fold_metrics(yte, proba, excess, direction)
    out["full"] = {"coverage": 1.0, "f1": full["f1"],
                   "precision": full["precision"], "recall": full["recall"]}
    for margin in margins:
        keep = np.abs(proba - 0.5) >= margin
        key = f"margin_{margin}"
        if keep.sum() == 0:
            out[key] = {"coverage": 0.0, "skipped": "no retained rows"}
            continue
        m = fold_metrics(yte[keep], proba[keep], excess[keep], direction[keep])
        out[key] = {
            "coverage": float(keep.mean()),
            "n_retained": int(keep.sum()),
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_gain_vs_full": float(m["f1"] - full["f1"]),
        }
    return out


def gate_verdict(paired: dict) -> dict:
    """Deterministic GO/NO-GO verdict from the pre-specified rule.

    Primary metric: mean PR-AUC across dev folds. Primary model:
    logreg_balanced. Primary comparison: C - A paired per fold.
    GO iff the mean paired difference > 0 AND the lower bound of the
    two-sided 95% Student-t CI (4 df) > 0.
    """
    prim = paired.get(PRIMARY_MODEL, {}).get("C-A/pr_auc", {})
    mean_diff = float(prim.get("mean_diff", float("nan")))
    ci = prim.get("ci95", [float("nan"), float("nan")])
    go = bool(
        np.isfinite(mean_diff) and mean_diff > 0 and np.isfinite(ci[0]) and ci[0] > 0
    )
    return {
        "rule": (
            "GO iff mean paired (C-A) PR-AUC difference > 0 AND the lower bound "
            "of the two-sided 95% Student-t CI (4 df) > 0; "
            f"primary model {PRIMARY_MODEL}, primary metric {PRIMARY_METRIC}"
        ),
        "primary_mean_diff": mean_diff,
        "primary_ci95": [float(ci[0]), float(ci[1])],
        "primary_t_pvalue": float(prim.get("t_pvalue", float("nan"))),
        "primary_wilcoxon_pvalue": float(prim.get("wilcoxon_pvalue", float("nan"))),
        "verdict": "GO" if go else "NO-GO",
        "test_period_touched": False,
    }


def run_benchmark(
    df: pd.DataFrame,
    trading_days: np.ndarray | None,
    label_cfg: LabelConfig | None = None,
    config: BenchmarkConfig | None = None,
    log=print,
) -> dict:
    """Phase 3 benchmark on the development period. Returns a JSON-safe
    results dict; the go/no-go verdict lives under ``gate``.

    ``df`` must be the ``build_dataset`` labeled frame (Phase 2 target and
    price columns required). ``trading_days`` is the union trading calendar
    (``load_trading_calendar_days``); it is REQUIRED (no default).
    """
    label_cfg = (label_cfg or LabelConfig()).validated()
    config = (config or BenchmarkConfig()).validated()
    if trading_days is None:
        raise ValueError("[phase3] trading_days is required; no default allowed")
    trading_days = np.asarray(trading_days)

    missing = [c for c in PRICE_COLS + ["excess_3d", "direction_3d"]
               if c not in df.columns]
    if missing:
        raise ValueError(f"[phase3] missing columns: {missing}")
    if "target_version" in df.columns:
        tv = df["target_version"].dropna().unique().tolist()
        log(f"[phase3] target set version: {tv}")

    work = df.dropna(subset=["published_at", "t0", "t1", "abn_ret"]).reset_index(
        drop=True
    )
    work = work.copy()
    if "text" not in work.columns or work["text"].isna().all():
        work["text"] = make_text(work)
    work["text"] = work["text"].fillna("").astype(str)
    point_in_time.check_frame(work, "run_benchmark")

    eff_dev_end = periods.DEV_END
    work = work[
        pd.to_datetime(work["t0"]).dt.date <= eff_dev_end
    ].reset_index(drop=True)
    if len(work) == 0:
        raise ValueError("[phase3] no development rows after dev-end filter")
    log(f"[phase3] dev rows: {len(work)} (t0 <= {eff_dev_end})")

    wf_cfg = splits.WalkForwardConfig(
        n_splits=config.n_splits,
        window=config.window,
        embargo_days=config.embargo_days,
        horizon_days=label_cfg.window_days,
        min_train=config.min_train,
    )
    wf_splits = splits.make_splits(work, wf_cfg, trading_days)

    fold_rows: list[dict] = []
    for i, sp in enumerate(wf_splits):
        tri, tei = sp["train_idx"], sp["test_idx"]
        # Per-fold attention, train-only quantiles: identical convention to
        # run_walk_forward (validation/labels.attention_keep_mask).
        fold_pos = np.concatenate([tri, tei])
        fold_df = work.iloc[fold_pos]
        train_df = work.iloc[tri]
        if label_cfg.attention_enabled:
            mask = vlabels.attention_keep_mask(
                train_df,
                fold_df,
                min_articles=label_cfg.attention_min_articles,
                top_quartile=label_cfg.attention_top_quartile,
            ).to_numpy()
        else:
            mask = np.ones(len(fold_df), dtype=bool)
        tri_k, tei_k = tri[mask[: len(tri)]], tei[mask[len(tri) :]]
        if len(tri_k) == 0 or len(tei_k) == 0:
            log(f"[phase3] fold {i+1}: attention filter emptied train or test block; skip")
            continue
        cutoff = vlabels.fold_cutoff(work["abn_ret"].iloc[tri_k], label_cfg.quantile)
        yte = vlabels.fold_labels(work["abn_ret"].iloc[tei_k], cutoff).to_numpy()
        if len(np.unique(yte)) < 2:
            log(f"[phase3] fold {i+1}: single-class test block; skip")
            continue

        if config.dedupe_policy == "first":
            tri_eff = dedupe.dedupe_first(work.iloc[tri_k]).index.to_numpy()
            sample_weight = None
        else:
            tri_eff = np.asarray(tri_k)
            sample_weight = dedupe.sample_weights(work.iloc[tri_eff])

        ytr = vlabels.fold_labels(work["abn_ret"].iloc[tri_eff], cutoff).to_numpy()
        excess_te = work["excess_3d"].iloc[tei_k].to_numpy(dtype=float)
        direction_te = work["direction_3d"].iloc[tei_k].to_numpy(dtype=float)

        mats_tr, mats_te, _ = _build_matrices(
            work["text"].iloc[tri_eff],
            work["text"].iloc[tei_k],
            work[PRICE_COLS].iloc[tri_eff],
            work[PRICE_COLS].iloc[tei_k],
        )
        fitted = _fit_variant_models(
            mats_tr, ytr, sample_weight, config.seed, config.model_names, log=log
        )

        fold: dict = {
            "fold": i + 1,
            "train_n": int(len(tri_eff)),
            "test_n": int(len(tei_k)),
            "cutoff": float(cutoff),
            "test_pos_rate": float(yte.mean()),
            "test_idx": [int(x) for x in tei_k],
            "metrics": {},
        }
        proba_store: dict[str, dict[str, np.ndarray]] = {}
        for variant in VARIANTS:
            fold["metrics"][variant] = {}
            proba_store[variant] = {}
            for name, fn in fitted[variant].items():
                p = np.asarray(fn(mats_te), dtype=float)
                proba_store[variant][name] = p
                fold["metrics"][variant][name] = fold_metrics(
                    yte, p, excess_te, direction_te
                )
        if config.calibrate and PRIMARY_MODEL in fitted["C"]:
            fold["calibration"] = _calibration_diagnostic(
                mats_tr, mats_te, ytr, yte, sample_weight, config.seed,
                config.min_calib_train, log=log,
            )
        if config.abstain and PRIMARY_MODEL in fitted["C"]:
            fold["abstention"] = _abstention_eval(
                proba_store["C"][PRIMARY_MODEL], yte, excess_te, direction_te,
                ABSTENTION_MARGINS,
            )
        fold_rows.append(fold)
        prim_ca = ""
        if PRIMARY_MODEL in fold["metrics"]["C"]:
            prim_ca = (
                f" C/A logreg_bal pr_auc="
                f"{fold['metrics']['C'][PRIMARY_MODEL]['pr_auc']:.4f}/"
                f"{fold['metrics']['A'][PRIMARY_MODEL]['pr_auc']:.4f}"
            )
        log(
            f"[phase3] fold {i+1}: train={len(tri_eff)} test={len(tei_k)} "
            f"pos={yte.mean():.3f}{prim_ca}"
        )

    if not fold_rows:
        raise ValueError("[phase3] no folds completed")

    first_model = next(
        name for name in config.model_names
        if name in fold_rows[0]["metrics"]["A"]
    )
    metric_names = list(fold_rows[0]["metrics"]["A"][first_model].keys())
    agg: dict = {}
    for variant in VARIANTS:
        agg[variant] = {}
        for name in config.model_names:
            if name not in fold_rows[0]["metrics"][variant]:
                continue
            agg[variant][name] = {
                m: _agg([f["metrics"][variant][name][m] for f in fold_rows])
                for m in metric_names
            }

    paired: dict = {}
    for name in config.model_names:
        if name not in fold_rows[0]["metrics"]["C"]:
            continue
        paired[name] = {}
        for m in ("pr_auc", "roc_auc", "brier", "f1"):
            paired[name][f"C-A/{m}"] = _paired_test(
                [f["metrics"]["C"][name][m] for f in fold_rows],
                [f["metrics"]["A"][name][m] for f in fold_rows],
            )
            paired[name][f"C-B/{m}"] = _paired_test(
                [f["metrics"]["C"][name][m] for f in fold_rows],
                [f["metrics"]["B"][name][m] for f in fold_rows],
            )

    gate = gate_verdict(paired)

    results = {
        "phase": 3,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "target_set_version": TARGET_SET_VERSION,
        "config": {
            "n_splits": config.n_splits,
            "window": config.window,
            "embargo_days": config.embargo_days,
            "min_train": config.min_train,
            "dedupe_policy": config.dedupe_policy,
            "seed": config.seed,
            "model_names": list(config.model_names),
        },
        "label_config": {
            "window_days": label_cfg.window_days,
            "quantile": label_cfg.quantile,
            "attention_enabled": label_cfg.attention_enabled,
            "attention_min_articles": label_cfg.attention_min_articles,
            "attention_top_quartile": label_cfg.attention_top_quartile,
        },
        "variants": VARIANT_DESCR,
        "price_columns": PRICE_COLS,
        "dev_rows": len(work),
        "n_folds_completed": len(fold_rows),
        "folds": fold_rows,
        "aggregate": agg,
        "paired": paired,
        "gate": gate,
    }
    return sanitize_json(results)

def _fmt(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return f"{x:.{nd}f}"


def write_gate_report(results: dict, path: str | Path) -> None:
    """Write docs/GATE.md: the pre-specified rule, the primary comparison,
    the full model x variant table, calibration/abstention diagnostics, and
    the deterministic GO/NO-GO verdict."""
    path = Path(path)
    gate = results["gate"]
    agg = results["aggregate"]
    paired = results["paired"]
    models = results["config"]["model_names"]

    lines: list[str] = []
    A = lines.append
    A("# Phase 3: Go / No-Go Gate")
    A("")
    A("## Pre-specified rule (frozen before the run)")
    A("")
    A(f"> {gate['rule']}")
    A("")
    A("- Primary comparison: variant C (text + price) vs variant A (price-only).")
    A("- Variants A/B/C share identical fold test indices; TF-IDF, the median")
    A("  imputer, and the scaler are fit on train data only, per fold.")
    A("- The test period (2026-07-01..2026-08-31) is untouched by this")
    A("  benchmark; it is evaluated at most once, and only after a GO verdict.")
    A("")
    A("## Primary comparison (C - A, paired per fold)")
    A("")
    prim = paired.get(PRIMARY_MODEL, {}).get("C-A/pr_auc", {})
    A(f"- Model: {PRIMARY_MODEL}; metric: mean PR-AUC over "
      f"{results['n_folds_completed']} dev folds.")
    A(f"- Mean paired difference: {_fmt(gate['primary_mean_diff'])}")
    ci = gate["primary_ci95"]
    A(f"- 95% CI: [{_fmt(ci[0])}, {_fmt(ci[1])}]")
    A(f"- Paired t p-value: {_fmt(gate['primary_t_pvalue'])}; "
      f"Wilcoxon p-value: {_fmt(gate['primary_wilcoxon_pvalue'])}")
    A("")
    A(f"## Verdict: {gate['verdict']}")
    A("")
    if gate["verdict"] == "GO":
        A("The gate rule fired GO: text adds signal over price-only on the")
        A("development walk-forward at the pre-specified bar. Next step: run")
        A("the single test-period evaluation")
        A("(`final_train_and_test(confirm_test_eval=True)`) once, append the")
        A("result to this document, then proceed to Phase 4.")
    else:
        A("The gate rule fired NO-GO: on the pre-specified bar, text does not")
        A("add signal over price-only (or the evidence is inconclusive).")
        A("Per the research program: STOP. Do not run the test-period")
        A("evaluation, do not build Phases 4-7, and report this negative")
        A("result to the user for an explicit decision before any further work.")
    A("")
    A("## Mean PR-AUC by model x variant (dev folds, 95% CI)")
    A("")
    A("| model | A (price) | B (text) | C (combined) | C-A diff [95% CI] |")
    A("|---|---|---|---|---|")
    for m in models:
        cells = []
        for v in ("A", "B", "C"):
            row = agg.get(v, {}).get(m, {}).get("pr_auc", {})
            mean, ci95 = row.get("mean", float("nan")), row.get("ci95", [float("nan")] * 2)
            cells.append(f"{_fmt(mean)} [{_fmt(ci95[0])}, {_fmt(ci95[1])}]")
        d = paired.get(m, {}).get("C-A/pr_auc", {})
        dc, dci = d.get("mean_diff", float("nan")), d.get("ci95", [float("nan")] * 2)
        cells.append(f"{_fmt(dc)} [{_fmt(dci[0])}, {_fmt(dci[1])}]")
        A(f"| {m} | {' | '.join(cells)} |")
    A("")
    A("## Calibration diagnostic (logreg_balanced, per variant, per fold mean)")
    A("")
    A("Platt (sigmoid) and isotonic recalibration fit inside the train block")
    A("only. This is diagnostic: PR-AUC is rank-based and unaffected by")
    A("monotone recalibration.")
    A("")
    cal_rows = []
    for f in results["folds"]:
        cal = f.get("calibration")
        if not cal or "skipped" in cal:
            continue
        cal_rows.append((f["fold"], cal))
    n_folds = results["n_folds_completed"]
    n_skip = n_folds - len(cal_rows)
    if n_skip:
        A(f"Calibration ran on {len(cal_rows)}/{n_folds} folds; {n_skip} "
          "fold(s) skipped (single-class TimeSeriesSplit(3) partitions: the "
          "train blocks are too small for three internal calibration splits).")
        A("")
    collapsed = [
        (fold_no, v, m)
        for fold_no, cal in cal_rows
        for v in VARIANTS
        for m in ("sigmoid", "isotonic")
        if v in cal and m in cal[v]
        and "proba_std" in cal[v][m] and cal[v][m]["proba_std"] < 1e-9
    ]
    if collapsed:
        A("Note: on the fold(s) where calibration ran, "
          + ", ".join(f"{m} ({v}, fold {n})" for n, v, m in collapsed)
          + " collapsed to a constant predictor (proba_std ~ 0). Any Brier "
          "improvement from a collapsed row is degenerate, not evidence of "
          "better calibration.")
        A("")
    if cal_rows:
        A("| variant | raw Brier | sigmoid Brier | isotonic Brier | raw ECE | "
          "sigmoid ECE | isotonic ECE |")
        A("|---|---|---|---|---|---|---|")
        for v in VARIANTS:
            def _m(rows, v=v, k1="raw", k2="brier"):
                return np.mean([r[1][v][k1][k2] for r in rows])
            A(f"| {v} | {_fmt(_m(cal_rows, k2='brier'))} | "
              f"{_fmt(_m(cal_rows, k1='sigmoid', k2='brier'))} | "
              f"{_fmt(_m(cal_rows, k1='isotonic', k2='brier'))} | "
              f"{_fmt(_m(cal_rows, k2='ece'))} | "
              f"{_fmt(_m(cal_rows, k1='sigmoid', k2='ece'))} | "
              f"{_fmt(_m(cal_rows, k1='isotonic', k2='ece'))} |")
    else:
        A("Calibration was skipped on every fold (train blocks below "
          "min_calib_train).")
    A("")
    A("## Abstention diagnostic (variant C, logreg_balanced, raw probas)")
    A("")
    A("Abstain when |p - 0.5| < margin. Kept only if measured to help "
      "(retained F1 > full F1).")
    A("")
    abs_rows = [(f["fold"], f["abstention"]) for f in results["folds"]
                if "abstention" in f]
    if abs_rows:
        A("| margin | mean coverage | mean retained F1 | mean F1 gain vs full |")
        A("|---|---|---|---|")
        for margin in ABSTENTION_MARGINS:
            k = f"margin_{margin}"
            cov = np.mean([r[1][k]["coverage"] for r in abs_rows
                           if "coverage" in r[1][k]])
            f1 = np.mean([r[1][k]["f1"] for r in abs_rows if "f1" in r[1][k]])
            gain = np.mean([r[1][k]["f1_gain_vs_full"] for r in abs_rows
                            if "f1_gain_vs_full" in r[1][k]])
            A(f"| {margin} | {_fmt(cov)} | {_fmt(f1)} | {_fmt(gain, 3)} |")
    else:
        A("Abstention was not evaluated (primary config absent).")
    A("")
    A("## Provenance")
    A("")
    A(f"- Generated (UTC): {results['generated_utc']}")
    A(f"- Target set version: {results['target_set_version']}")
    A(f"- Dev rows: {results['dev_rows']}; folds completed: "
      f"{results['n_folds_completed']}")
    A(f"- Label config: window_days="
      f"{results['label_config']['window_days']}, quantile="
      f"{results['label_config']['quantile']}, attention_enabled="
      f"{results['label_config']['attention_enabled']}, min_articles="
      f"{results['label_config']['attention_min_articles']}, top_quartile="
      f"{results['label_config']['attention_top_quartile']}")
    A(f"- Config: {json.dumps(results['config'])}")
    A("- Price columns: "
      f"{', '.join(results['price_columns'])}")
    A("")
    path.write_text("\n".join(lines))
    print(f"[phase3] gate report -> {path}")


def main() -> None:
    from signal_lab.models.build_and_train import (
        build_dataset,
        load_trading_calendar_days,
    )

    df, _ = build_dataset(log=print)
    trading_days = load_trading_calendar_days(log=print)
    results = run_benchmark(df, trading_days, log=print)
    ART.mkdir(parents=True, exist_ok=True)
    out_json = ART / "phase3_benchmark.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[phase3] results -> {out_json}")
    write_gate_report(results, REPO_ROOT / "docs" / "GATE.md")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    parser = argparse.ArgumentParser(description="Phase 3 benchmark and gate")
    parser.parse_args()
    main()
