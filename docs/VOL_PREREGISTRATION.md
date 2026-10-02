# VOL Pre-registration: volatility prediction campaign

**Status:** PRE-REGISTERED. This document is committed BEFORE any
volatility experiment is run. Any change to the frozen target, baseline,
models, horizon, metric, or periods after this commit is logged as a
deviation in `docs/VOL_RESULTS.md`.

**Context:** The direction-prediction research program closed NO-GO at
gate 1 (docs/NEGATIVE_RESULT.md) and the gate-2 widening was blocked by
GDELT throttling. The extraction program (APP-000..APP-006) hardened the
explanatory workbench. This is a NEW, SEPARATELY pre-registered campaign
on a different target: volatility prediction. It is not a continuation of
the direction bet and does not reuse its conclusions. The test period
(2026-07-01..2026-08-31) and the confirmation period (t0 >= 2026-09-01)
remain untouched until the pre-registered evaluation point described
below.

## Frozen target

- **Target:** `realized_vol_5d`, versioned in `TargetConfig` target
  version 2.0 (docs/TARGETS_AND_FEATURES.md): sample std (ddof=1) of daily
  log returns over the 5 trading days (t0, t0+5], decimal, daily.
- **Frame:** one row per (ticker, trading day). Rows whose 5-day window
  is partial (fewer than 5 forward trading days with returns) have NaN
  targets and are DROPPED, never filled. This is a regression task; the
  per-article attention frame of the direction program is not used.
- **Forecast origin:** features use only information knowable at t0
  (returns through t0's close, prices through t0's close). The label
  window (t0, t0+5] never enters a feature.

## Frozen periods

- **Development:** t0 <= 2026-06-30. All selection, fitting, and
  walk-forward happen here.
- **Test:** 2026-07-01..2026-08-31. Untouched during development. The
  single authorized test evaluation runs only after the dev campaign
  reaches GO (see decision rule).
- **Confirmation:** t0 >= 2026-09-01. NEVER read, scored, or inspected,
  for any purpose including debugging. `periods.check_no_confirmation`
  remains the enforcement point.

## Universe and prices (dev)

- **Universe:** S&P 100 constituents as of the pre-campaign date, plus
  SPY as the market leg. Exact ticker list committed as
  `data/universe_vol.csv` in the build step; count reported.
- **Prices:** yfinance daily OHLCV, 2022-01-01 through 2026-07-15. The
  2022 start gives the trailing 252-day GARCH estimation window and the
  22-day HAR monthly leg a full burn-in before the first dev row.
  Download gaps are logged explicitly and never backfilled silently.
- **Data-quality gates** (fail loudly): duplicate (ticker, date) rows,
  timestamp ordering violations, missing closes for labeled rows,
  zero-volume stretches inconsistent with the exchange calendar.

## Baseline and arms

- **naive (sanity check):** forecast = trailing `realized_vol_5d`
  (persistence). Reported, no gate authority.
- **garch11 (BASELINE):** GARCH(1,1) fit by maximum likelihood on the
  trailing 252 trading days of daily log returns through t0. Forecast =
  analytic 5-day term structure:
  E[sigma^2_{t+h}|F_t] = sigma_bar^2 + (alpha+beta)^(h-1) *
  (sigma^2_{t+1|t} - sigma_bar^2), summed over h = 1..5, square-rooted
  to realized vol. Fit failures (non-convergence, alpha+beta >= 1) fall
  back to the naive forecast and are counted, never hidden.
- **Price ML arms (PRIMARY):** ridge regression and LightGBM regressor
  on HAR-style point-in-time price features: RV_{t-1}, RV_{t-5},
  RV_{t-22} (Corsi HAR daily/weekly/monthly legs), lagged squared
  returns, Parkinson high-low range volatility, log volume and volume
  changes, day-of-week, and the same legs for SPY. Feature availability
  declared per `FEATURE_POINT_IN_TIME` conventions.
- **Text arms (SECONDARY, exploratory):** the selected price arm's
  features plus the TF-IDF + Loughran-McDonald lexicon text features
  (vectorizer fit on train texts only per fold), aggregated to
  ticker-day. Reported with the same paired tests but with NO gate
  authority. A significant text win becomes a finding for a follow-up
  campaign, not a GO.

## Selection honesty

- Arm selection (ridge vs LightGBM, hyperparameters) runs on a SINGLE
  purged train/validation split inside development, per the extraction
  program's item-6 harness. The walk-forward folds are for evaluation,
  never selection.
- The Deflated Sharpe Ratio (extraction item 6) is reported as a
  model-selection honesty metric. It does not gate the campaign.

## Validation

- 5-fold expanding purged walk-forward on t0, seed 7, min_train 50.
- **Purge:** drop train rows whose [t0, t0+5] label window overlaps the
  current fold's test range (j == k only, per the documented
  `make_splits` deviation note).
- **Embargo:** 5 trading days after each test fold (>= horizon).
- The splitter asserts its invariants on every call (no train/test
  overlap, no label-window overlap among kept rows, embargo zones empty).

## Primary metric

- **QLIKE** (Patton 2011): QLIKE(sigma^2, sigma_hat^2) =
  sigma^2/sigma_hat^2 - ln(sigma^2/sigma_hat^2) - 1. QLIKE is consistent
  for ranking volatility forecasts and robust to noise in the realized
  proxy. Lower is better.
- Secondary (reported, no gate authority): MSE on variance, MAE on
  volatility, Diebold-Mariano test of the QLIKE loss differential vs
  the GARCH baseline.

## Decision rule (frozen)

- **Primary comparison:** the selected price-arm champion vs garch11,
  paired per dev walk-forward fold on QLIKE.
- **Statistic:** mean paired QLIKE differential d = QLIKE(garch11) -
  QLIKE(champion) across folds; positive means the champion beats GARCH.
- **GO** iff mean(d) > 0 AND the lower bound of the 95% two-sided
  Student-t CI on d (df = n_folds - 1) is > 0. Otherwise NO-GO. A narrow
  miss is NO-GO; moving the bar after seeing results is forbidden.
- On GO: run the single authorized test-period evaluation
  (`confirm_test_eval=True`), report dev and test QLIKE side by side,
  then stop. Further work needs a new pre-registration.
- On NO-GO: stop, write `docs/VOL_NEGATIVE_RESULT.md`. No peeking at
  test or confirmation, no target switching inside this campaign.

## Kill criterion (frozen)

- If the primary comparison is NO-GO, the campaign is DEAD. It is not
  revived by trying other models, other features, other horizons, or
  other metrics. Any of those is a new campaign with its own
  pre-registration.
- The pre-registered next target is **event magnitude** (|excess_3d|,
  regression) or **drawdown risk**, as a SEPARATELY pre-registered
  campaign started only after VOL's negative result is written. Nothing
  about VOL's data, features, or tuning carries a gate claim into the
  next campaign.

## Deviation policy

Any deviation from this document after its commit (target, baseline,
models, horizon, metric, universe, periods, rule) is logged in
`docs/VOL_RESULTS.md` (or `docs/VOL_NEGATIVE_RESULT.md` on NO-GO) under
"Deviations", with rationale. Undocumented deviations invalidate the
pre-registration.
