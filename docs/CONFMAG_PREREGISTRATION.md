# CONFMAG Pre-registration: conformal-abstention-gated event-magnitude prediction

**Status:** PRE-REGISTERED. This document is committed BEFORE any
CONFMAG experiment is run. Any change to the frozen target, model,
conformal construction, selection rule, baseline, metric, or periods
after this commit is logged as a deviation in the result document.

**Context:** This is BET 3 (SPECULATIVE) from
`docs/SYNTHESIS_BETS.md`: magnitude may be predictable only in a subset
of regimes, and the model itself can detect those regimes via
conformal interval width. This is selective prediction ("regression
with a reject option," cf. Geifman and El-Yaniv) applied to event
magnitude in finance, which the surveyed literature does not do. The
VOL, HARVIX, EVENTVOL, TAILQ, DISPVOL, and ATTRVOL campaigns are all
dead (NO-GO). This is a NEW, SEPARATELY pre-registered campaign, and
the LAST in the program. Nothing about their results carries a gate
claim into this campaign except the frozen universe/frame definitions.

The test period (2026-07-01..2026-08-31) and the confirmation period
(t0 >= 2026-09-01) remain untouched until the pre-registered
evaluation point described below.

## Frozen target

- **Target:** `|excess_3d|(t0)` = absolute 3-day abnormal return:
  excess_3d(t0) = sum_{h=1..3} (r_i(t0+h) - r_SPY(t0+h)), where r are
  daily log returns from split/dividend-adjusted closes; the target
  is the absolute value, decimal.
- **Frame:** one row per (ticker, trading day), ALL ticker-days (not
  event-restricted). Rows whose 3-day label window (t0, t0+3] is
  partial are DROPPED, never filled. Features use only information
  knowable at t0's close. The label window never enters a feature.

## Frozen features

- The 34 HAR+VIX features, EXACTLY the HARVIX definitions (31 HAR
  price features + vix_level, vix_5d_change, vix_slope). No new
  features: the novelty under test is the conformal gating, not the
  feature set. The feature frame is built by reusing the proven
  VOL/HARVIX/DISPVOL pipeline; only the target column is new.

## Arms

- **constant (PRIMARY BASELINE):** per fold, the train-block mean of
  |excess_3d|. This is the honest bar the research sprint names for
  magnitude targets (memo section 2.3): no published strong baseline
  exists for absolute abnormal returns.
- **garch_implied (SECONDARY BASELINE):** sqrt(3/5) * garch_vol_5d *
  sqrt(2/pi), where garch_vol_5d is the scale-corrected 5-day GARCH
  term-structure forecast (daily scale) from the frozen VOL
  implementation. Rationale: under the GARCH forecast the 3-day
  abnormal return is approximately N(0, sigma_3d^2) with
  sigma_3d = sqrt(3/5) * garch_vol_5d, and E|X| = sigma*sqrt(2/pi)
  for X ~ N(0, sigma^2). This is an approximation (it ignores the
  term structure shape and the SPY leg), documented here as such.
  Reported, no gate authority.
- **lgbm_gated (CHALLENGER):** LightGBM L2 regressor with the frozen
  VOL hyperparameters (n_estimators=300, learning_rate=0.05,
  num_leaves=31, min_child_samples=100, feature_fraction=0.8,
  bagging_fraction=0.8, lambda_l2=1.0, random_state=7, verbose=-1)
  on the 34 HAR+VIX features, trained per fold on the fold's FULL
  train block. Its MSE is evaluated ONLY on the selected subset
  (see selection rule). Forecasts are floored at 0 (magnitudes are
  non-negative; the floor replaces the 1e-4 vol floor).
- The comparison constant-vs-challenger is on the SAME selected
  subset, so the selection cannot manufacture an edge by itself.

## Conformal construction (frozen): Conformalized Quantile Regression

Standard split-conformal with absolute residuals yields
constant-width intervals, which would make width-based selection
degenerate. This campaign therefore uses Conformalized Quantile
Regression (Romano, Patterson, Candes 2019), which yields adaptive
widths, implemented in the repo's typed style reusing the
finite-sample corrected quantile from
`signal_lab.models.conformal.conformal_quantile`:

- Per fold, the train block is split TEMPORALLY: the most recent 20%
  of train rows (by t0) form the calibration block; the rest is the
  proper-train block.
- Fit two LightGBM quantile regressors on proper-train at
  tau_lo = 0.1 and tau_hi = 0.9 (frozen LGBM hyperparameters above
  plus objective='quantile', alpha=tau).
- On the calibration block, conformity scores
  E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)).
- q_hat = conformal_quantile(E, alpha=0.2) = the
  ceil((n_cal + 1) * 0.8) / n_cal order statistic (the finite-sample
  correction that gives the 80% marginal coverage guarantee).
- Test intervals: [q_lo(x) - q_hat, q_hi(x) + q_hat]; widths vary by
  row. Empirical coverage on test is reported as a diagnostic
  (should sit near 80%).

## Selection rule (frozen)

- Calibration interval widths: w_i = (q_hi(x_i) - q_lo(x_i)) +
  2 * q_hat on the calibration block. (Ranking by conformalized
  width equals ranking by raw quantile width since q_hat is constant
  per fold; the conformalized form is used for reporting.)
- Threshold t* = 25th percentile of calibration widths.
- SELECT the test rows with conformalized interval width <= t*.
  The threshold is computed from calibration data only, before any
  test label is seen.
- Selection rate = (total selected test rows across folds) /
  (total test rows across folds).

## Primary metric

- **MSE** on |excess_3d| over the SELECTED test rows (lower is
  better). Paired per fold: d_k = MSE(constant) - MSE(challenger)
  on fold k's selected rows.
- Secondary (reported, no gate authority): MAE on the selected
  subset; full-test-set (unselected) MSE for both arms; selection
  rate; empirical interval coverage; Diebold-Mariano on the MSE
  differential; 95% Model Confidence Set; DSR honesty metric.

## Decision rule (frozen, conjunctive)

- **GO** iff ALL of the following hold:
  (1) mean(d) > 0 across folds, AND
  (2) the lower bound of the 95% two-sided Student-t CI on d
      (df = n_folds - 1) is > 0, AND
  (3) the pooled selection rate is in [0.10, 0.40].
  Otherwise NO-GO. A narrow miss on any conjunct is NO-GO; moving
  the bar after seeing results is forbidden.
- The selection-rate guard rules out degenerate empty selections
  (which would make MSE undefined) and near-universal selection
  (which would make the "gating" vacuous).
- On GO: run the single authorized test-period evaluation
  (`confirm_test_eval=True`), report dev and test MSE side by side,
  then stop. Further work needs a new pre-registration.
- On NO-GO: stop, write `docs/CONFMAG_NEGATIVE_RESULT.md`. No peeking
  at test or confirmation, no target switching inside this campaign.

## Kill criterion (frozen)

- If the conjunctive rule is NO-GO, the campaign is DEAD. It is not
  revived by trying other models, other features, other quantiles,
  other selection thresholds, or other metrics. Any of those is a new
  campaign with its own pre-registration.
- CONFMAG is the LAST pre-registered campaign in the program. On its
  NO-GO, the program reports all six campaigns exhausted with no GO.

## Validation

- 5-fold expanding purged walk-forward on t0, seed 7, min_train 50.
- **Purge:** drop train rows whose [t0, t0+3] label window overlaps
  the current fold's test range (j == k only, per the documented
  `make_splits` convention). The horizon here is 3 trading days to
  match the 3-day label window.
- **Embargo:** 3 trading days after each test fold (>= horizon).
- The splitter asserts its invariants on every call (no train/test
  overlap, no label-window overlap among kept rows, embargo zones
  empty).

## Selection honesty

- The estimators and the selection rule are frozen: this is a
  one-arm gating test, not a model search. No selection split is run
  beyond the frozen calibration split; the 25th-percentile threshold
  is frozen by this document.
- The Deflated Sharpe Ratio (extraction item 6) is reported as a
  model-selection honesty metric: per-row MSE gains of the
  challenger over the constant baseline as pseudo-returns, freq=1,
  pooled dev selected test rows. It does not gate the campaign.

## Frozen periods

- **Development:** t0 <= 2026-06-30. All fitting, calibration, and
  walk-forward happen here.
- **Test:** 2026-07-01..2026-08-31. Untouched during development. The
  single authorized test evaluation runs only after a dev GO (see
  decision rule).
- **Confirmation:** t0 >= 2026-09-01. NEVER read, scored, or
  inspected, for any purpose including debugging.
  `periods.check_no_confirmation` remains the enforcement point on
  every frame build, split, and evaluation call.

## Universe and data (dev)

- **Universe:** the frozen `data/universe_vol.csv`: S&P 100 (101) +
  SPY = 102 tickers. No change from VOL/HARVIX/DISPVOL/ATTRVOL.
- **Prices:** yfinance daily OHLCV, 2022-01-01 through 2026-07-15
  (same download as VOL; same quality gates; download gaps logged,
  never backfilled silently).
- **VIX:** ^VIX and ^VIX3M daily closes, same series and burn-in rules
  as HARVIX (cached locally, uncommitted; fail-loud download).
- **NO GDELT anywhere in this campaign.** All features are
  price-derived.

## Deviation policy

Any deviation from this document after its commit (target, model,
conformal construction, selection rule, baseline, metric, universe,
periods, rule) is logged in `docs/CONFMAG_RESULTS.md` (or
`docs/CONFMAG_NEGATIVE_RESULT.md` on NO-GO) under "Deviations", with
rationale. Undocumented deviations invalidate the pre-registration.

## Why it might fail (stated upfront, from the bet memo)

Conformal width may track feature-space density rather than
predictability (tight intervals where data is dense, which need not
be where magnitude is forecastable); the selected subset may be too
small for the CI to clear zero; the constant baseline may already be
near-optimal for absolute returns, which are notoriously close to
unpredictable.
