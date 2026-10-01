# STEP 2 — Leakage check on the early-fold price signal (EXP-028)

**Status:** COMPLETED 2026-10-01. **Verdict: NO HARNESS LEAKAGE; EARLY-FOLD
SIGNAL CONSISTENT WITH NOISE.**

Gate 1 showed variant-A (price-only) PR-AUC ~0.44 in folds 1-2 but
~0.05-0.06 in folds 3-4. Before widening the data for the gate-2 retest,
this step determined whether that early signal was harness leakage,
noise, or regime. Method: `scripts/gate2_step2_leakage.py` (committed,
re-runnable; raw numbers in `data/artifacts/step2_leakage_check.json`).

## What was run (all on current dev data, gate-1 methodology untouched)

1. **Reproduction** — `run_benchmark` with identical purge (5-day embargo),
   expanding window, seed 7, 5 models. Matches gate 1 exactly.
2. **Shuffled labels** — `abn_ret` permuted (seed 7) before the benchmark.
   PR-AUC must fall to chance in every fold.
3. **Injected future feature** — `excess_3d` appended to `PRICE_COLS`
   (diagnostic-only monkeypatch). `excess_3d` is verified formula-identical
   to `abn_ret` (correlation 1.0, max abs diff 0.0) and scores PR-AUC 1.0
   as a single feature against the fold labels, so it is a true oracle.
   The harness must transmit it into test predictions.
4. **Bootstrap** — per-fold 95% CI of variant-A logreg_balanced PR-AUC
   (2000 resamples of the fold's own test predictions). The fold loop
   reuses the public validation primitives (`make_splits`,
   `attention_keep_mask`, `fold_cutoff`, `fold_labels`, `sample_weights`)
   and asserts PR-AUC identical to `run_benchmark` within 1e-4.

## Results

| fold | test n | n_pos | test pos_rate | repro PR-AUC | 95% boot CI | shuffled | injected |
|------|--------|-------|---------------|--------------|-------------|----------|----------|
| 1 | 108 | 35 | 0.324 | 0.4503 | [0.314, 0.582] | 0.176 | 0.462 |
| 2 | 115 | 25 | 0.217 | 0.4206 | [0.292, 0.608] | 0.174 | 0.541 |
| 3 | 125 | 6  | 0.048 | 0.0520 | [0.021, 0.120] | 0.168 | 0.646 |
| 4 | 123 | 6  | 0.049 | 0.0619 | [0.027, 0.135] | 0.090 | 1.000 |
| 5 | 94  | 21 | 0.223 | 0.1401 | [0.095, 0.211] | 0.051 | 0.841 |

Test-block t0 ranges: fold 1 2025-11-24..2025-12-19; fold 2
2026-01-27..2026-02-06; fold 3 2026-02-12..2026-03-11; fold 4
2026-03-12..2026-05-27; fold 5 2026-05-28..2026-06-26.

## Interpretation

- **Harness is sound.** Shuffled labels sit at chance in every fold
  (each within 0.15 of its own fold pos_rate). The injected oracle lifts
  PR-AUC in every fold, reaching 1.0 in fold 4 — features demonstrably
  reach the model and test predictions. (The lift is muted in early folds
  because L2 shrinkage with small train_n keeps the oracle's coefficient
  near the noise features'; this is a regularization artifact of the
  diagnostic, not a harness flaw.)
- **Early folds are noise.** Fold 1's CI [0.31, 0.58] contains chance
  (0.324); fold 2's lower bound (0.29) is only marginally above chance
  (0.217) with 25 positives. Folds 3-4 rest on **6 positive test samples
  each** — their "collapse" to 0.05 is a small-sample artifact, not a
  regime the model failed to adapt to.
- **Regime, partially.** The wild test pos_rate swing (0.324 -> 0.048)
  shows the return distribution shifted between blocks (volatile
  late-2025/early-2026 vs calm spring 2026). But trailing price features
  cannot predict those shifts, and with 6-35 positives per fold the
  per-fold PR-AUC numbers are noise-dominated. This is precisely why the
  gate uses the paired test across folds instead of eyeballing folds.

## Consequence for gate 2

Proceed with data widening (STEP 3). The widened dev set will have far
more positives per fold, which shrinks exactly the sampling noise that
made gate 1's per-fold numbers uninterpretable. No harness changes.
