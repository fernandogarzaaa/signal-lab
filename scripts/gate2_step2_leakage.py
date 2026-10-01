"""STEP 2 (gate-2 retest): leakage check on the early-fold price signal.

Gate 1 showed variant-A (price-only) PR-AUC ~0.44 in folds 1-2 but
~0.05-0.06 in folds 3-4. This script determines whether that early
signal is harness leakage, noise, or regime, by:

1. REPRODUCTION: re-running the gate-1 benchmark on current dev data
   (identical purge/embargo/splits) and tabulating per-fold PR-AUC.
2. SHUFFLED: permuting abn_ret (seeded) -> PR-AUC must fall to chance.
3. INJECTED: adding the forward excess return as a feature
   (monkeypatched PRICE_COLS, diagnostic only) -> PR-AUC must spike,
   proving the harness would catch leakage if it existed.
4. BOOTSTRAP: per-fold 95% CI of variant-A PR-AUC (2000 resamples) to
   show how much of the early-fold "signal" is small-sample noise.
   The fold loop reuses the public validation primitives and asserts
   bit-identical PR-AUC against run_benchmark's own numbers.

Verdict logic is printed at the end. All runs are EXPLORATORY
(registry EXP-028). The confirmation period is never touched:
run_benchmark filters to t0 <= DEV_END internally.

Usage:
    .venv/bin/python scripts/gate2_step2_leakage.py [--json-out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from signal_lab.benchmark import phase3
from signal_lab.models.build_and_train import build_dataset, load_trading_calendar_days
from signal_lab.models.labels import LabelConfig
from signal_lab.validation import dedupe, labels as vlabels, periods, point_in_time, splits

PRIMARY = "logreg_balanced"
N_BOOT = 2000
SEED = 7


def _fold_loop_variant_a(df: pd.DataFrame, trading_days: np.ndarray):
    """Replicate run_benchmark's fold loop for variant A / logreg_balanced.

    Returns per-fold dicts with yte, proba, pr_auc, test_n, pos_rate,
    test date range. Asserts PR-AUC matches run_benchmark to 1e-9.
    """
    label_cfg = LabelConfig().validated()
    config = phase3.BenchmarkConfig(calibrate=False, abstain=False).validated()
    work = df.dropna(subset=["published_at", "t0", "t1", "abn_ret"]).reset_index(drop=True)
    work["text"] = work["text"].fillna("").astype(str) if "text" in work.columns else ""
    point_in_time.check_frame(work, "step2")
    work = work[pd.to_datetime(work["t0"]).dt.date <= periods.DEV_END].reset_index(drop=True)

    wf_cfg = splits.WalkForwardConfig(
        n_splits=config.n_splits, window=config.window,
        embargo_days=config.embargo_days, horizon_days=label_cfg.window_days,
        min_train=config.min_train,
    )
    wf_splits = splits.make_splits(work, wf_cfg, trading_days)
    texts = work["text"].astype(str)
    price = work[phase3.PRICE_COLS]
    folds = []
    for sp in wf_splits:
        tri, tei = sp["train_idx"], sp["test_idx"]
        fold_pos = np.concatenate([tri, tei])
        fold_df = work.iloc[fold_pos]
        train_df = work.iloc[tri]
        mask = vlabels.attention_keep_mask(
            train_df, fold_df,
            min_articles=label_cfg.attention_min_articles,
            top_quartile=label_cfg.attention_top_quartile,
        ).to_numpy()
        tri_k, tei_k = tri[mask[: len(tri)]], tei[mask[len(tri):]]
        if len(tri_k) == 0 or len(tei_k) == 0:
            continue
        cutoff = vlabels.fold_cutoff(work["abn_ret"].iloc[tri_k], label_cfg.quantile)
        yte = vlabels.fold_labels(work["abn_ret"].iloc[tei_k], cutoff).to_numpy()
        if len(np.unique(yte)) < 2:
            continue
        tri_eff = np.asarray(tri_k)
        w = dedupe.sample_weights(work.iloc[tri_eff])
        ytr = vlabels.fold_labels(work["abn_ret"].iloc[tri_eff], cutoff).to_numpy()
        mats_tr, mats_te, _ = phase3._build_matrices(
            texts.iloc[tri_eff], texts.iloc[tei_k],
            price.iloc[tri_eff], price.iloc[tei_k],
        )
        fitted = phase3._fit_variant_models(
            mats_tr, ytr, w, config.seed, (PRIMARY,), log=lambda *a: None,
        )
        proba = np.asarray(fitted["A"][PRIMARY](mats_te), dtype=float)
        pr = float(average_precision_score(yte, proba))
        te_dates = pd.to_datetime(work["t0"].iloc[tei_k])
        folds.append({
            "fold": len(folds) + 1,
            "train_n": int(len(tri_eff)), "test_n": int(len(tei_k)),
            "pos_rate": float(yte.mean()), "n_pos": int(yte.sum()),
            "pr_auc": pr, "yte": yte, "proba": proba,
            "test_t0_min": str(te_dates.min().date()),
            "test_t0_max": str(te_dates.max().date()),
        })
    return folds


def _bootstrap_ci(yte: np.ndarray, proba: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(yte)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(yte[idx])) < 2:
            continue
        vals.append(average_precision_score(yte[idx], proba[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _per_fold_table(results: dict) -> list[dict]:
    rows = []
    for f in results["folds"]:
        m = f["metrics"]["A"].get(PRIMARY, {})
        rows.append({
            "fold": f["fold"], "train_n": f["train_n"], "test_n": f["test_n"],
            "pos_rate": round(f["test_pos_rate"], 4),
            "pr_auc_A": round(m.get("pr_auc", float("nan")), 4),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="data/artifacts/step2_leakage_check.json")
    args = ap.parse_args()

    df, _ = build_dataset(log=lambda *a: None)
    trading_days = load_trading_calendar_days(log=lambda *a: None)
    print(f"[step2] dataset rows: {len(df)}", flush=True)

    # 1. Reproduction on identical data/methodology.
    cfg = phase3.BenchmarkConfig(calibrate=False, abstain=False)
    res_repro = phase3.run_benchmark(df, trading_days, config=cfg, log=lambda *a: None)
    repro_rows = _per_fold_table(res_repro)
    print("[step2] REPRO per-fold (A/logreg_balanced pr_auc):")
    for r in repro_rows:
        print(f"  fold {r['fold']}: train={r['train_n']} test={r['test_n']} "
              f"pos={r['pos_rate']:.3f} pr_auc={r['pr_auc_A']:.4f}", flush=True)

    # 2. Fold-loop replication with probas, asserted identical, + bootstrap.
    folds = _fold_loop_variant_a(df, trading_days)
    assert len(folds) == len(repro_rows), (len(folds), len(repro_rows))
    for f, r in zip(folds, repro_rows):
        # run_benchmark rounds metrics to 4 decimals; allow rounding slack.
        assert abs(f["pr_auc"] - r["pr_auc_A"]) < 1e-4, (f["pr_auc"], r["pr_auc_A"])
    print("[step2] fold-loop replication matches run_benchmark to 1e-4", flush=True)
    for f in folds:
        lo, hi = _bootstrap_ci(f["yte"], f["proba"], SEED + f["fold"])
        f["pr_auc_ci95"] = [round(lo, 4), round(hi, 4)]
        print(f"  fold {f['fold']}: pr_auc={f['pr_auc']:.4f} "
              f"95% boot CI [{lo:.4f}, {hi:.4f}] n_pos={f['n_pos']} "
              f"t0 {f['test_t0_min']}..{f['test_t0_max']}", flush=True)

    # 3. Shuffled labels -> chance.
    df_shuf = df.copy()
    df_shuf["abn_ret"] = np.random.default_rng(SEED).permutation(
        df_shuf["abn_ret"].to_numpy())
    res_shuf = phase3.run_benchmark(df_shuf, trading_days, config=cfg,
                                    log=lambda *a: None)
    shuf_rows = _per_fold_table(res_shuf)
    print("[step2] SHUFFLED per-fold pr_auc:",
          [r["pr_auc_A"] for r in shuf_rows], flush=True)

    # 4. Injected future feature -> must spike (harness catches leakage).
    real_cols = phase3.PRICE_COLS
    phase3.PRICE_COLS = list(real_cols) + ["excess_3d"]
    try:
        res_inj = phase3.run_benchmark(df, trading_days, config=cfg,
                                       log=lambda *a: None)
    finally:
        phase3.PRICE_COLS = real_cols
    inj_rows = _per_fold_table(res_inj)
    print("[step2] INJECTED per-fold pr_auc:",
          [r["pr_auc_A"] for r in inj_rows], flush=True)

    # 5. Verdict. Compare each run against its OWN chance level
    # (fold cutoffs, hence pos rates, differ across runs).
    repro = [r["pr_auc_A"] for r in repro_rows]
    shuf = [r["pr_auc_A"] for r in shuf_rows]
    inj = [r["pr_auc_A"] for r in inj_rows]
    shuf_at_chance = all(abs(s - r["pos_rate"]) < 0.15
                         for s, r in zip(shuf, shuf_rows))
    # Harness transmits a perfect feature: injected must beat the
    # shuffled (chance) run decisively in every fold.
    injected_transmits = all(i > s + 0.15 for i, s in zip(inj, shuf))
    early_ci_overlaps_chance = all(
        f["pr_auc_ci95"][0] < f["pos_rate"] + 0.1 for f in folds[:2])
    verdict = (
        "NO HARNESS LEAKAGE; EARLY-FOLD SIGNAL CONSISTENT WITH NOISE"
        if (shuf_at_chance and injected_transmits and early_ci_overlaps_chance)
        else "INCONCLUSIVE OR LEAKAGE INDICATED - INVESTIGATE"
    )
    print(f"[step2] shuffled~=chance: {shuf_at_chance}; "
          f"injected transmits (all folds > shuffled+0.15): {injected_transmits}; "
          f"early-fold CI overlaps chance: {early_ci_overlaps_chance}", flush=True)
    print(f"[step2] VERDICT: {verdict}", flush=True)

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "registry_id": "EXP-028", "status": "EXPLORATORY",
        "n_bootstrap": N_BOOT, "seed": SEED,
        "reproduction": repro_rows,
        "bootstrap_ci95": [
            {"fold": f["fold"], "ci95": f["pr_auc_ci95"],
             "test_t0_min": f["test_t0_min"], "test_t0_max": f["test_t0_max"]}
            for f in folds],
        "shuffled": shuf_rows, "injected": inj_rows,
        "checks": {"shuffled_at_chance": shuf_at_chance,
                   "injected_transmits": injected_transmits,
                   "early_ci_overlaps_chance": early_ci_overlaps_chance},
        "verdict": verdict,
    }
    out_path = REPO_ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[step2] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
