# DISPVOL campaign results: HAR+VIX+dispersion vs HAR+VIX (working document)

Pre-registration: `docs/DISPVOL_PREREGISTRATION.md` (frozen, binding). This document is the working record; the results PR carries the final version.

## Configuration

- Challenger: **har_vix_disp** = ridge (alpha=1.0, standardized) on the 34 frozen HAR+VIX features + 6 frozen dispersion features (disp_level, disp_5d_change, disp_z60, disp_level_x_rv1d, disp_5d_change_x_rv1d, disp_z60_x_rv1d)
- Primary baseline: **har_vix** = ridge (alpha=1.0, standardized) on the 34 HAR+VIX features alone (the HARVIX challenger definition, unchanged)
- Secondary baseline: **garch11** (scale-corrected 5-day term structure; fit failures fall back to naive persistence)
- Sanity arm: naive (trailing realized_vol_5d)
- Validation: 5-fold expanding purged walk-forward on t0, seed 7, min_train 50, purge j==k, 5-trading-day embargo
- Primary metric: QLIKE on variance (lower is better); 95% MCS reported, not a gate
- Dev rows: 113511 (107067 after NaN-feature drop)
- GARCH naive fallbacks on test rows: 1666 / 89167

## Per-fold QLIKE (variance)

| fold | test range | n_train | n_test | purged | embargoed | QLIKE har_vix_disp | QLIKE har_vix | QLIKE garch11 | QLIKE naive | d = harvix - challenger | d2 = garch - challenger |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-12-20..2023-09-06 | 17400 | 17800 | 500 | 0 | 0.572268 | 0.570694 | 0.563762 | 1.698339 | -0.001574 | -0.008506 |
| 2 | 2023-09-07..2024-05-22 | 35200 | 17918 | 500 | 0 | 0.902833 | 0.901771 | 0.624821 | 2.167574 | -0.001062 | -0.278012 |
| 3 | 2024-05-23..2025-02-05 | 52613 | 17776 | 505 | 500 | 0.759259 | 0.761347 | 0.696948 | 2.267553 | +0.002087 | -0.062311 |
| 4 | 2025-02-06..2025-10-17 | 69884 | 17925 | 505 | 1005 | 15.845790 | 13.089495 | 0.735470 | 1.841576 | -2.756295 | -15.110320 |
| 5 | 2025-10-20..2026-06-30 | 87299 | 17748 | 510 | 1510 | 0.579349 | 0.607108 | 0.516799 | 1.771612 | +0.027759 | -0.062550 |

## Verdict computation (frozen decision rule)

- Primary: d_k = QLIKE(har_vix) - QLIKE(har_vix_disp) per fold: -0.001574, -0.001062, +0.002087, -2.756295, +0.027759
- mean(d) = -0.545817, se = 0.552646, df = 4.0, t(0.975) = 2.7764
- 95% two-sided Student-t CI on mean(d): [-2.080209, +0.988575]
- Diebold-Mariano on fold-level QLIKE differentials (paired-t form, HAC-equivalent on non-overlapping blocks): stat = -0.9876, two-sided p = 0.3792, n = 5.0
- GO iff mean(d) > 0 AND CI lower bound > 0: **NO-GO**

## Secondary comparison: challenger vs garch11 (reported, no gate authority)

- d2_k = QLIKE(garch11) - QLIKE(har_vix_disp) per fold: -0.008506, -0.278012, -0.062311, -15.110320, -0.062550
- mean(d2) = -3.104340, 95% CI [-11.438817, +5.230137]
- DM: stat = -1.0341, p = 0.3595

## Model Confidence Set (95%, reported, not a gate)

- Survivors: garch11 (n_rows=89167, B=5000, alpha=0.05, seed=7)
- Eliminated (in order): har_vix_disp (p=0.0002), har_vix (p=0.0002), naive (p=0.0002)

## Secondary metrics (per fold, no gate authority)

| fold | MSE-var challenger | MSE-var har_vix | MSE-var garch11 | MAE-vol challenger | MAE-vol har_vix | MAE-vol garch11 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.00000039 | 0.00000039 | 0.00000036 | 0.006874 | 0.006838 | 0.007288 |
| 2 | 0.00000055 | 0.00000055 | 0.00000054 | 0.005579 | 0.005582 | 0.006525 |
| 3 | 0.00000079 | 0.00000079 | 0.00000112 | 0.007008 | 0.007011 | 0.007416 |
| 4 | 0.00000152 | 0.00000152 | 0.00000148 | 0.007778 | 0.007791 | 0.009184 |
| 5 | 0.00000108 | 0.00000108 | 0.00000113 | 0.007823 | 0.007702 | 0.007898 |

## Selection honesty (DSR, reported, not a gate)

per-row QLIKE gains of the tried arms over naive as pseudo-returns, freq=1, pooled dev test rows; reported only, not a gate

- har_vix: Sharpe = -0.0056, DSR = 0.0053, likely_false_discovery = True
- har_vix_disp: Sharpe = -0.0071, DSR = 0.0002, likely_false_discovery = True

## Deviations

None.
