# Phase 3: Go / No-Go Gate

## Pre-specified rule (frozen before the run)

> GO iff mean paired (C-A) PR-AUC difference > 0 AND the lower bound of the two-sided 95% Student-t CI (4 df) > 0; primary model logreg_balanced, primary metric pr_auc

- Primary comparison: variant C (text + price) vs variant A (price-only).
- Variants A/B/C share identical fold test indices; TF-IDF, the median
  imputer, and the scaler are fit on train data only, per fold.
- The test period (2026-07-01..2026-08-31) is untouched by this
  benchmark; it is evaluated at most once, and only after a GO verdict.

## Primary comparison (C - A, paired per fold)

- Model: logreg_balanced; metric: mean PR-AUC over 5 dev folds.
- Mean paired difference: 0.0035
- 95% CI: [-0.0091, 0.0160]
- Paired t p-value: 0.4828; Wilcoxon p-value: 1.0000

## Verdict: NO-GO

The gate rule fired NO-GO: on the pre-specified bar, text does not
add signal over price-only (or the evidence is inconclusive).
Per the research program: STOP. Do not run the test-period
evaluation, do not build Phases 4-7, and report this negative
result to the user for an explicit decision before any further work.

## Mean PR-AUC by model x variant (dev folds, 95% CI)

| model | A (price) | B (text) | C (combined) | C-A diff [95% CI] |
|---|---|---|---|---|
| naive | 0.1723 [0.0223, 0.3223] | 0.1723 [0.0223, 0.3223] | 0.1723 [0.0223, 0.3223] | 0.0000 [0.0000, 0.0000] |
| historical_rate | 0.1723 [0.0223, 0.3223] | 0.1723 [0.0223, 0.3223] | 0.1723 [0.0223, 0.3223] | 0.0000 [0.0000, 0.0000] |
| logreg_plain | 0.2240 [-0.0164, 0.4644] | 0.2142 [0.0126, 0.4157] | 0.2277 [-0.0203, 0.4757] | 0.0037 [-0.0086, 0.0160] |
| logreg_balanced | 0.2250 [-0.0176, 0.4676] | 0.2079 [0.0156, 0.4001] | 0.2285 [-0.0207, 0.4776] | 0.0035 [-0.0091, 0.0160] |
| lightgbm_balanced | 0.2734 [-0.1137, 0.6604] | 0.1925 [0.1109, 0.2742] | 0.2444 [-0.1455, 0.6343] | -0.0290 [-0.0690, 0.0111] |

## Calibration diagnostic (logreg_balanced, per variant, per fold mean)

Platt (sigmoid) and isotonic recalibration fit inside the train block
only. This is diagnostic: PR-AUC is rank-based and unaffected by
monotone recalibration.

Calibration ran on 1/5 folds; 4 fold(s) skipped (single-class TimeSeriesSplit(3) partitions: the train blocks are too small for three internal calibration splits).

Note: on the fold(s) where calibration ran, sigmoid (A, fold 1), isotonic (A, fold 1), sigmoid (B, fold 1), isotonic (B, fold 1), sigmoid (C, fold 1), isotonic (C, fold 1) collapsed to a constant predictor (proba_std ~ 0). Any Brier improvement from a collapsed row is degenerate, not evidence of better calibration.

| variant | raw Brier | sigmoid Brier | isotonic Brier | raw ECE | sigmoid ECE | isotonic ECE |
|---|---|---|---|---|---|---|
| A | 0.2800 | 0.2429 | 0.3241 | 0.2893 | 0.1543 | 0.3241 |
| B | 0.2533 | 0.2429 | 0.3241 | 0.2082 | 0.1543 | 0.3241 |
| C | 0.2779 | 0.2429 | 0.3241 | 0.2927 | 0.1543 | 0.3241 |

## Abstention diagnostic (variant C, logreg_balanced, raw probas)

Abstain when |p - 0.5| < margin. Kept only if measured to help (retained F1 > full F1).

| margin | mean coverage | mean retained F1 | mean F1 gain vs full |
|---|---|---|---|
| 0.05 | 0.9459 | 0.1639 | -0.015 |
| 0.1 | 0.8923 | 0.1636 | -0.015 |
| 0.15 | 0.8538 | 0.1225 | -0.056 |

## Provenance

- Generated (UTC): 2026-10-01T11:02:45.641882+00:00
- Target set version: 2.0
- Dev rows: 693; folds completed: 5
- Label config: window_days=3, quantile=0.9, attention_enabled=True, min_articles=2, top_quartile=True
- Config: {"n_splits": 5, "window": "expanding", "embargo_days": 5, "min_train": 50, "dedupe_policy": "weight", "seed": 7, "model_names": ["naive", "historical_rate", "logreg_plain", "logreg_balanced", "lightgbm_balanced"]}
- Price columns: ctx_vol20_pct, ctx_mom5, ctx_mom20, ctx_sector_rel5, ctx_src_tier, ctx_word_count, ctx_hour_bucket, px_ret_1d, px_rsi_14, px_macd_hist, px_ma_dist_20, px_ma_dist_50, px_realvol_5d, px_realvol_20d, px_atr_14, px_vol_z_20, px_relvol_20, px_vs_spy_5d, px_vs_spy_20d, mkt_spy_ret_20d, mkt_spy_vol_20d
