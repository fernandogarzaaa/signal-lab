# Research Note: Signal Lab 0.3.0 — Honest Evaluation of a News-Sentiment Classifier

**Date:** 2026-10-01
**Dataset:** 806 labeled articles (baseline scheme), 2025-09-30 to 2026-09-29
**Code:** `signal_lab.models.walk_forward`, `signal_lab.models.calibration`
**Artifacts:** `data/artifacts/walk_forward.json`, `data/artifacts/calibration.json`

## Summary

We rebuilt the evaluation harness for our news-sentiment classifier around
three questions a quant team would actually ask: (1) does performance hold up
across time, or was the single split lucky? (2) do the predicted probabilities
mean what they say? (3) can we abstain on low-confidence articles to improve
the trade-off? The answers are: (1) mostly not — regime instability dominates;
(2) roughly, and Platt scaling helps while isotonic hurts; (3) no — the
confidence gate removes coverage without improving precision or backtest
returns. We report all three, including the negative results, because a model
card that only reports wins is a marketing document.

## 1. Purged walk-forward cross-validation

Single time-based splits hide regime instability. We implemented 5-fold
expanding-window walk-forward CV with explicit purge and embargo:

- **Purge:** training rows whose label window could overlap the test period
  are removed.
- **Embargo:** a gap of 5 trading days between train end and test start.
- **No leakage in featurization:** the TF-IDF vocabulary is fit on training
  rows only, per fold.
- Fold 1 was unscorable (no positives in its early training window) and is
  skipped loudly, not silently.

Results (baseline label scheme, per-fold PR-AUC):

| fold | test period | n_train | n_test | pos rate | PR-AUC | F1 |
|------|------------|---------|--------|----------|--------|-----|
| 1 | — | — | — | — | skipped | skipped |
| 2 | … | 205 | 134 | 0.112 | 0.1431 | 0.0000 |
| 3 | … | 316 | 134 | 0.149 | 0.2553 | 0.0714 |
| 4 | … | 430 | 135 | 0.037 | 0.0478 | 0.0000 |
| 5 | … | 543 | 134 | 0.179 | 0.2335 | 0.0000 |

Aggregate: mean PR-AUC **0.170** (std 0.095, 95% CI [0.019, 0.321]) vs mean
baseline (positive rate) 0.119 (95% CI [0.022, 0.217]). Mean F1 at the default
0.5 threshold: **0.018** (95% CI [-0.039, 0.075]) — effectively zero.

Interpretation: the single-split PR-AUC of ~0.16 hid major fold-to-fold
variation (0.048 to 0.255). The model's average ranking beats the average
prevalence baseline, but the confidence interval is extremely wide and F1 at
the operating threshold is nil. We do not claim the model is strong,
profitable, or statistically significant. The model didn't get smarter; the
dataset got bigger and the evaluation got more honest.

## 2. Probability calibration

On a temporal 60/15/25 train/calibrate/test split (202 test articles):

| variant | ECE | PR-AUC | F1 |
|---------|-----|--------|-----|
| raw | 0.1103 | 0.1629 | 0.0476 |
| Platt | **0.0765** | 0.1629 | 0.0000 |
| isotonic | 0.1345 | 0.1379 | 0.0000 |

Platt scaling improves calibration (ECE 0.110 → 0.077) without changing
ranking. Isotonic regression *hurts* calibration (ECE → 0.135): with only 121
calibration points at ~10% positive rate, the nonparametric fit overfits. We
report this rather than tuning the bin count until isotonic looks good.
Calibration also shrinks probabilities toward the base rate, which is why F1
at 0.5 drops to zero — an honest calibrated model mostly says "unlikely."

## 3. Confidence abstention

**Classification:** acting only when P(positive) >= tau, for tau in
[0.10, 0.15, 0.20, 0.30, 0.50]:

| tau | coverage | n kept | precision |
|-----|----------|--------|-----------|
| 0.10 | 0.87 | 175 | 0.067 |
| 0.20 | 0.48 | 97 | 0.067 |
| 0.50 | 0.07 | 15 | 0.067 |

Precision is flat at 0.067 — below the 0.101 base rate — across the entire
threshold range. Higher model confidence does not mean higher precision. The
gate does not separate.

**Backtest:** `--min-confidence 0.2` keeps 97 of 902 scored articles, but the
event rule (>= 2 articles per ticker-day at the p95 trigger) collapses to 4
events. Return 0.0075 vs 0.0107 ungated; Sharpe 0.814 vs 0.792; paired p =
0.30. The gate removes coverage without improving risk-adjusted returns.

## 4. What we changed in the data pipeline along the way

While building the walk-forward harness we found that 878 of 1,263 articles
had a missing sector-relative-strength feature: the SPY fallback series only
covered 2026-07-02 onward, so the benchmark leg was absent for earlier dates
and stage 3 silently dropped those rows (385 context-complete articles
remained). We now neutralize the missing leg to 0.0 ("no measured
sector-relative signal") and log the count, instead of silently discarding
69% of the data. Labeled rows went from 385 to 806. This is disclosed, not
hidden, because it changes every downstream number.

## 5. Bottom line

The 0.3.0 evaluation stack — purged walk-forward CV with embargo, ECE +
reliability diagrams, per-fold confidence intervals, abstention curves wired
into both classification and backtest coverage — is the durable contribution.
The model itself remains a weak ranker of noisy weak labels. That is the
honest result, and the harness is built to keep it honest as the data and
models improve.
