# HARVIX campaign results: HAR+VIX rematch (working document)

Pre-registration: `docs/HARVIX_PREREGISTRATION.md` (frozen, binding). This document is the working record; the results PR carries the final version.

## Configuration

- Challenger: **har_vix** = ridge (alpha=1.0, standardized) on the 31 VOL HAR features + 3 frozen VIX features (vix_level, vix_5d_change, vix_slope)
- Primary baseline: garch11 (scale-corrected 5-day term structure; fit failures fall back to naive persistence)
- Secondary baseline: har = ridge (alpha=1.0, standardized) on the 31 HAR features alone
- Sanity arm: naive (trailing realized_vol_5d)
- Validation: 5-fold expanding purged walk-forward on t0, seed 7, min_train 50, purge j==k, 5-trading-day embargo
- Primary metric: QLIKE on variance (lower is better); 95% MCS reported, not a gate
- Dev rows: 113511 (111267 after NaN-feature drop)
- GARCH naive fallbacks on test rows: 5166 / 92667

## Per-fold QLIKE (variance)

| fold | test range | n_train | n_test | purged | embargoed | QLIKE har_vix | QLIKE garch11 | QLIKE har | QLIKE naive | d = garch - harvix | d2 = har - harvix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-10-31..2023-07-27 | 18100 | 18500 | 500 | 0 | 0.570555 | 0.696149 | 0.545860 | 1.599077 | +0.125595 | -0.024694 |
| 2 | 2023-07-28..2024-04-23 | 36600 | 18600 | 500 | 0 | 0.740193 | 0.600372 | 0.626063 | 2.041675 | -0.139821 | -0.114130 |
| 3 | 2024-04-24..2025-01-15 | 54700 | 18480 | 500 | 500 | 0.711194 | 0.657060 | 0.740367 | 2.172974 | -0.054134 | +0.029173 |
| 4 | 2025-01-16..2025-10-08 | 72673 | 18625 | 505 | 1002 | 4.882932 | 0.785199 | 0.794081 | 2.043785 | -4.097733 | -4.088851 |
| 5 | 2025-10-09..2026-06-30 | 90788 | 18462 | 510 | 1507 | 0.614739 | 0.519340 | 0.609344 | 1.759039 | -0.095399 | -0.005395 |

## Verdict computation (frozen decision rule)

- Primary: d_k = QLIKE(garch11) - QLIKE(har_vix) per fold: +0.125595, -0.139821, -0.054134, -4.097733, -0.095399
- mean(d) = -0.852298, se = 0.812610, df = 4.0, t(0.975) = 2.7764
- 95% two-sided Student-t CI on mean(d): [-3.108466, +1.403869]
- Diebold-Mariano on fold-level QLIKE differentials (paired-t form, HAC-equivalent on non-overlapping blocks): stat = -1.0488, two-sided p = 0.3534, n = 5.0
- GO iff mean(d) > 0 AND CI lower bound > 0: **NO-GO**

## Secondary comparison: har_vix vs har (reported, no gate authority)

- d2_k = QLIKE(har) - QLIKE(har_vix) per fold: -0.024694, -0.114130, +0.029173, -4.088851, -0.005395
- mean(d2) = -0.840780, 95% CI [-3.096260, +1.414701]
- DM: stat = -1.0350, p = 0.3591

## Model Confidence Set (95%, reported, not a gate)

- Survivors: har_vix, garch11, har (n_rows=92667, B=5000, alpha=0.05, seed=7)
- Eliminated (in order): naive (p=0.0002)

## Secondary metrics (per fold, no gate authority)

| fold | MSE-var har_vix | MSE-var garch11 | MSE-var har | MAE-vol har_vix | MAE-vol garch11 | MAE-vol har |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.00000038 | 0.00000047 | 0.00000038 | 0.006193 | 0.007853 | 0.007380 |
| 2 | 0.00000051 | 0.00000051 | 0.00000049 | 0.005543 | 0.006524 | 0.006143 |
| 3 | 0.00000058 | 0.00000085 | 0.00000058 | 0.006759 | 0.007146 | 0.006659 |
| 4 | 0.00000165 | 0.00000167 | 0.00000259 | 0.007846 | 0.009273 | 0.008647 |
| 5 | 0.00000105 | 0.00000112 | 0.00000106 | 0.007682 | 0.007927 | 0.007574 |

## Selection honesty (DSR, reported, not a gate)

per-row QLIKE gains of the tried arms over naive as pseudo-returns, freq=1, pooled dev test rows; reported only, not a gate

- har: Sharpe = +0.0983, DSR = 1.0000, likely_false_discovery = False
- har_vix: Sharpe = +0.0036, DSR = 0.0000, likely_false_discovery = True

## Deviations

None.
