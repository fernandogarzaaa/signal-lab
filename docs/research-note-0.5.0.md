# Research Note 0.5.0: Honest Re-evaluation and Negative Results

## Summary

0.5.0 is a honesty release. We re-ran the 0.4.0 experiments with exact
official parameters and found the FinBERT win is real but smaller than
reported: **0.141 PR-AUC** (not 0.150). We tested four improvement directions;
all were negative or noise-level. The binding constraint remains data and
label quality.

## Corrected Baseline

The 0.4.0 report claimed FinBERT+ctx achieved 0.1502 mean PR-AUC. Re-running
with the exact official walk-forward parameters (purge=3d, embargo=3d) gives:

- FinBERT+ctx (LogReg): **0.1409** mean PR-AUC
  - Fold 1: 0.1435, Fold 2: 0.0949, Fold 3: 0.1625, Fold 4: 0.0775, Fold 5: 0.2262
- TF-IDF+lexicon+ctx: 0.1219 (from 0.4.0)

The FinBERT win holds (+0.019, +16% relative), but the magnitude was overstated
in 0.4.0. We correct the record here.

## Negative Results (First-Class)

### 1. HistGradientBoosting: Appears to win, but overfits

HGB on FinBERT+ctx achieved 0.1812 mean PR-AUC (+0.04 over LogReg), winning
3 of 5 folds. However:

- Predicted probabilities were severely miscalibrated (max 0.0044)
- F1 at 0.5 threshold was 0.0 across all folds
- Platt calibration **destroyed** the gain: PR-AUC dropped to 0.1195

Conclusion: HGB overfits to the ranking. The probabilities don't generalize.
**Do not use.** We reverted to LogReg.

### 2. Sentiment Dispersion: No signal

Hypothesis: Disagreement across articles (high std of sentiment) predicts
volatility or reversals. LSEG has per-article sentiment but not cross-article
dispersion.

Result: r=0.0075, p=0.92 (n=174 ticker-days). **Pure noise.** Documented and
abandoned.

### 3. Sentiment Momentum: Weak, not significant

Hypothesis: 3-day sentiment slope predicts next-day return (momentum or
reversal). LSEG doesn't provide multi-day sentiment trend.

Result: r=-0.10, p=0.09 (n=274). Negative sign suggests reversal, but not
significant. **Inconclusive, not shipped.**

### 4. LogReg Hyperparameter Tuning: Noise-level

- Best C=0.1: 0.1438 (+0.003 over C=1.0)
- Best class_weight=None: 0.1423 (+0.001 over balanced)

Gains are within noise. Model is already well-tuned. **No change.**

## Positive: Ctx Feature Ablation

The 7 context features collectively add +0.0064 PR-AUC (0.1409 vs 0.1345
FinBERT-only). Leave-one-out:

- Without ctx_hour_bucket: 0.1339 (-0.0070) **most important**
- Without ctx_vol20_pct: 0.1390 (-0.0019)
- Without ctx_mom20: 0.1395 (-0.0014)
- Without ctx_sector_rel5: 0.1409 (0.0000) **no effect**
- Without ctx_mom5: 0.1418 (+0.0009) *noise*
- Without ctx_src_tier: 0.1426 (+0.0017) *noise*
- Without ctx_word_count: 0.1418 (+0.0009) *noise*

Hour of publication matters (likely captures pre-market vs intraday effects).
Sector relative strength has zero effect in this dataset.

## Data Quality: Good News

The 0.4.0 report feared 25% of texts lacked FinBERT embeddings (204/813).
Re-checking: **100% coverage** on the 902 labeled rows (813 unique texts).
The missing embeddings were in the unlabeled set. No data quality issue for
the model.

## What's Next

The binding constraints are unchanged:

1. **Data quantity**: 902 rows, 90 positives. GDELT expansion blocked by 429s.
2. **Label quality**: Weak labels from price movements. Inan's gold-set labeling
   (150 articles) will quantify the noise directly.

No model architecture will fix these. The next real improvement needs more
data or better labels.
