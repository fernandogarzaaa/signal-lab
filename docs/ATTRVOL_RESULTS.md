# ATTRVOL campaign results: HAR+VIX+attribution vs HAR+VIX on event days (working document)

Pre-registration: `docs/ATTRVOL_PREREGISTRATION.md` (frozen, binding). This document is the working record; the results PR carries the final version.

## Configuration

- Challenger: **har_vix_attr** = ridge (alpha=1.0, standardized) on the 34 frozen HAR+VIX features + 5 frozen attribution features (driver_H, idio_share, ret_x_H, analogue_vol_mean, analogue_vol_std)
- Primary baseline: **har_vix** = ridge (alpha=1.0, standardized) on the 34 HAR+VIX features alone (the HARVIX challenger definition, unchanged)
- Secondary baseline: **garch11** (scale-corrected 5-day term structure; fit failures fall back to naive persistence)
- Sanity arm: naive (trailing realized_vol_5d)
- Event rows: trailing top-decile |abn_ret| per ticker (|abn_ret| >= trailing 252d 90th percentile, min 60 obs)
- Validation: 5-fold expanding purged walk-forward on t0, seed 7, min_train 50, purge j==k, 5-trading-day embargo; fold stats over event-day test rows
- Primary metric: QLIKE on variance (lower is better); 95% MCS reported, not a gate
- Dev event rows: 11688 (11688 after NaN-feature drop)
- GARCH naive fallbacks on test rows: 114 / 9733

## Per-fold QLIKE (variance, event-day test rows)

| fold | test range | n_train | n_test | purged | embargoed | QLIKE har_vix_attr | QLIKE har_vix | QLIKE garch11 | QLIKE naive | d = harvix - challenger | d2 = garch - challenger |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2023-03-15..2024-02-07 | 1902 | 1942 | 53 | 0 | 0.701677 | 0.605741 | 0.553045 | 0.821799 | -0.095936 | -0.148632 |
| 2 | 2024-02-08..2024-09-11 | 3828 | 1955 | 69 | 0 | 49.543414 | 49.088867 | 0.697457 | 1.188429 | -0.454547 | -48.845957 |
| 3 | 2024-09-12..2025-04-09 | 5724 | 1978 | 77 | 51 | 0.835467 | 0.807458 | 0.875990 | 1.217647 | -0.028008 | +0.040523 |
| 4 | 2025-04-10..2026-01-22 | 7537 | 1916 | 205 | 88 | 0.707180 | 0.612610 | 0.575006 | 0.899718 | -0.094570 | -0.132174 |
| 5 | 2026-01-23..2026-06-30 | 9530 | 1942 | 58 | 158 | 0.692147 | 0.651433 | 0.490048 | 0.783441 | -0.040713 | -0.202099 |

## Verdict computation (frozen decision rule)

- Primary: d_k = QLIKE(har_vix) - QLIKE(har_vix_attr) per fold: -0.095936, -0.454547, -0.028008, -0.094570, -0.040713
- mean(d) = -0.142755, se = 0.079154, df = 4.0, t(0.975) = 2.7764
- 95% two-sided Student-t CI on mean(d): [-0.362522, +0.077012]
- Diebold-Mariano on fold-level QLIKE differentials (paired-t form, HAC-equivalent on non-overlapping blocks): stat = -1.8035, two-sided p = 0.1456, n = 5.0
- GO iff mean(d) > 0 AND CI lower bound > 0: **NO-GO**

## Secondary comparison: challenger vs garch11 (reported, no gate authority)

- d2_k = QLIKE(garch11) - QLIKE(har_vix_attr) per fold: -0.148632, -48.845957, +0.040523, -0.132174, -0.202099
- mean(d2) = -9.857668, 95% CI [-36.920115, +17.204779]
- DM: stat = -1.0113, p = 0.3691

## Model Confidence Set (95%, reported, not a gate)

- Survivors: garch11 (n_rows=9733, B=5000, alpha=0.05, seed=7)
- Eliminated (in order): har_vix_attr (p=0.0002), har_vix (p=0.0002), naive (p=0.0002)

## Secondary metrics (per fold, no gate authority)

| fold | MSE-var challenger | MSE-var har_vix | MSE-var garch11 | MAE-vol challenger | MAE-vol har_vix | MAE-vol garch11 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.00000029 | 0.00000037 | 0.00000036 | 0.005919 | 0.006041 | 0.007356 |
| 2 | 0.00000119 | 0.00000118 | 0.00000375 | 0.007324 | 0.007250 | 0.007803 |
| 3 | 0.00000203 | 0.00000204 | 0.00000280 | 0.008865 | 0.009180 | 0.010008 |
| 4 | 0.00000194 | 0.00000184 | 0.00000199 | 0.008169 | 0.008051 | 0.009898 |
| 5 | 0.00000142 | 0.00000145 | 0.00000195 | 0.008529 | 0.008643 | 0.009203 |

## Selection honesty (DSR, reported, not a gate)

per-row QLIKE gains of the tried arms over naive as pseudo-returns, freq=1, pooled dev event test rows; reported only, not a gate

- har_vix: Sharpe = -0.0135, DSR = 0.0009, likely_false_discovery = True
- har_vix_attr: Sharpe = -0.0137, DSR = 0.0006, likely_false_discovery = True

## Deviations

None.
