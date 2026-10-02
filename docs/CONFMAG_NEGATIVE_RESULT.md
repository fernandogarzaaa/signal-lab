# CONFMAG campaign: NEGATIVE RESULT (NO-GO, campaign dead)

**Pre-registration:** `docs/CONFMAG_PREREGISTRATION.md` (frozen, binding;
merged as PR #74, commit 05abc15). This is BET 3 (SPECULATIVE) from
`docs/SYNTHESIS_BETS.md`: event magnitude may be predictable only in a
subset of regimes, and the model can detect those regimes via conformal
interval width. Target: `|excess_3d|` = absolute 3-day abnormal return
(sum of daily log returns over (t0, t0+3] minus the SPY leg), decimal, on
ALL ticker-days (not event-restricted). Challenger: **lgbm_gated** =
LightGBM L2 regressor (frozen VOL hyperparameters: n_estimators=300,
learning_rate=0.05, num_leaves=31, min_child_samples=100,
feature_fraction=0.8, bagging_fraction=0.8, lambda_l2=1.0,
random_state=7) on the 34 frozen HAR+VIX features, scored ONLY on the
CQR-selected subset. Primary baseline: **constant** = per-fold train
mean of |excess_3d|, scored on the same selected subset. Secondary
baseline: **garch_implied** = sqrt(3/5) * garch_vol_5d * sqrt(2/pi)
(reported, no gate authority). Conformal: CQR with LightGBM quantile
regressors at tau=0.1/0.9, alpha=0.2, temporal 20% calibration split per
fold; selection = test rows with conformalized interval width <= 25th
percentile of calibration widths. Metric: MSE on |excess_3d| over the
selected subset (lower is better). Validation: 5-fold expanding purged
walk-forward on t0, seed 7, min_train 50, 3-day horizon purge, 3-day
embargo. Universe: S&P 100 (101) + SPY = 102 tickers, frozen in
`data/universe_vol.csv`. Prices: yfinance 2022-01-01..2026-07-15
(reused VOL download; quality gates passed). VIX: ^VIX + ^VIX3M (reused
HARVIX cache). No GDELT anywhere in this campaign.

## Verdict: NO-GO

The frozen conjunctive decision rule: GO iff ALL of (1) mean(d) > 0,
(2) the lower bound of the 95% two-sided Student-t CI on d
(df = n_folds - 1) > 0, (3) the pooled selection rate is in
[0.10, 0.40], where d_k = MSE(constant) - MSE(challenger) on fold k's
selected rows. Otherwise NO-GO.

### Per-fold MSE (|excess_3d|, selected subset)

| fold | test range | n_train | n_test | selected | sel rate | coverage | MSE challenger | MSE constant | MSE garch_implied | d = const - challenger |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-10-31..2023-07-27 | 18117 | 18315 | 6180 | 0.337 | 0.827 | 0.000257 | 0.000272 | 0.000285 | +0.000016 |
| 2 | 2023-07-28..2024-04-23 | 36432 | 18414 | 5531 | 0.300 | 0.803 | 0.000192 | 0.000233 | 0.000226 | +0.000040 |
| 3 | 2024-04-24..2025-01-16 | 54549 | 18397 | 2432 | 0.132 | 0.793 | 0.000225 | 0.000233 | 0.000297 | +0.000007 |
| 4 | 2025-01-17..2025-10-08 | 72646 | 18342 | 3704 | 0.202 | 0.791 | 0.000315 | 0.000315 | 0.000382 | +0.000000 |
| 5 | 2025-10-09..2026-06-30 | 90685 | 18281 | 3300 | 0.181 | 0.765 | 0.000281 | 0.000274 | 0.000382 | -0.000007 |

- mean(d) = +0.000011; 95% CI = [-0.000011, +0.000034]. Mean positive
  but the lower bound is below zero, so conjunct (2) fails. **NO-GO.**
- Diebold-Mariano on the differential: stat +1.3990, p = 0.2344.
- Pooled selection rate = 21147 / 91749 = 0.2305, inside the guard
  [0.10, 0.40] (conjunct (3) passes).
- Empirical interval coverage: 0.827 / 0.803 / 0.793 / 0.791 / 0.765,
  near the nominal 80%: the CQR construction is calibrated; the
  selection is not degenerate.
- 95% Model Confidence Set (reported, not a gate): survivors =
  challenger; eliminated garch_implied (p=0.0002), constant
  (p=0.0006). The MCS prefers the challenger, but the frozen rule is
  conjunctive and the CI lower bound fails, so the verdict stands.

## Deviations

- **DSR honesty metric not computed.** `dsr_report` requires at least
  2 tried variants; CONFMAG is a one-arm gating test (no model search),
  so exactly one variant was tried and the metric's selection-bias
  correction is vacuous. The runner skipped it fail-loud instead of
  crashing (PR #78), and the skip is recorded in
  `docs/confmag_results.json` under "deviations". The metric does not
  gate the campaign, so the verdict is unaffected.

## What this means

The conformal gate worked as designed (calibrated intervals, sane
selection rates), but the gated subset did not deliver a statistically
significant MSE edge over the constant baseline. Magnitude on the
selected subset remains dominated by the baseline. The bet's own
upfront failure mode materialized: conformal width tracks
feature-space density / calibration geometry, not forecastability.

## Frozen periods

The test period (2026-07-01..2026-08-31) was never evaluated. The
confirmation period (t0 >= 2026-09-01) was never read, scored, or
inspected. CONFMAG is the LAST pre-registered campaign in the program:
the program is exhausted and further work needs the user's explicit
new direction.
