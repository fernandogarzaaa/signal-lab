# CONFMAG campaign results: conformal-gated |excess_3d| vs constant (working document)

Pre-registration: `docs/CONFMAG_PREREGISTRATION.md` (frozen, binding). This document is the working record; the results PR carries the final version.

## Configuration

- Challenger: **lgbm_gated** = LightGBM L2 regressor (frozen VOL hyperparameters) on the 34 frozen HAR+VIX features, scored ONLY on the CQR-selected subset
- Primary baseline: **constant** = per-fold train mean |excess_3d|, scored on the same selected subset
- Secondary baseline: **garch_implied** = sqrt(3/5) * garch_vol_5d * sqrt(2/pi)
- Conformal: CQR, LightGBM tau=0.1/0.9, alpha=0.2, temporal 20% calibration split per fold; selection = width <= 25th percentile of calibration widths
- Validation: 5-fold expanding purged walk-forward on t0, seed 7, min_train 50, 3-day horizon purge, 3-day embargo
- Primary metric: MSE on |excess_3d| over the selected subset
- Dev rows: 112385 (110163 after NaN-feature drop)
- Selected: 21147 / 91749 test rows (rate 0.2305, guard [0.10, 0.40])
- GARCH naive fallbacks on test rows: 5106 / 91749

## Per-fold MSE (|excess_3d|, selected subset)

| fold | test range | n_train | n_test | selected | sel rate | coverage | MSE challenger | MSE constant | MSE garch_implied | d = const - challenger |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-10-31..2023-07-27 | 18117 | 18315 | 6180 | 0.337 | 0.827 | 0.000257 | 0.000272 | 0.000285 | +0.000016 |
| 2 | 2023-07-28..2024-04-23 | 36432 | 18414 | 5531 | 0.300 | 0.803 | 0.000192 | 0.000233 | 0.000226 | +0.000040 |
| 3 | 2024-04-24..2025-01-16 | 54549 | 18397 | 2432 | 0.132 | 0.793 | 0.000225 | 0.000233 | 0.000297 | +0.000007 |
| 4 | 2025-01-17..2025-10-08 | 72646 | 18342 | 3704 | 0.202 | 0.791 | 0.000315 | 0.000315 | 0.000382 | +0.000000 |
| 5 | 2025-10-09..2026-06-30 | 90685 | 18281 | 3300 | 0.181 | 0.765 | 0.000281 | 0.000274 | 0.000382 | -0.000007 |

## Verdict computation (frozen conjunctive rule)

- Primary: d_k = MSE(constant) - MSE(challenger) per fold: +0.000016, +0.000040, +0.000007, +0.000000, -0.000007
- mean(d) = +0.000011, 95% two-sided Student-t CI: [-0.000011, +0.000034]
- Diebold-Mariano: stat = +1.3990, p = 0.2344
- Pooled selection rate = 0.2305 (guard [0.10, 0.40])
- GO iff mean(d) > 0 AND CI lower > 0 AND rate in guard: **NO-GO**

## Model Confidence Set (95%, reported, not a gate)

- Survivors: challenger
- Eliminated (in order): garch_implied (p=0.0002), constant (p=0.0006)

## Deviations

- DSR skipped: dsr_report needs at least 2 tried variants; single-variant campaign, no selection bias to deflate
