# DISPVOL Pre-registration: cross-sectional dispersion predicts individual realized volatility

**Status:** PRE-REGISTERED. This document is committed BEFORE any
DISPVOL experiment is run. Any change to the frozen target, features,
arms, horizon, metric, or periods after this commit is logged as a
deviation in the result document.

**Context:** This is BET 1 (SPECULATIVE) from
`docs/SYNTHESIS_BETS.md`: disagreement models (Harris and Raviv 1993;
Banerjee and Kremer) predict that wider belief dispersion produces
more trading and larger price moves, so dispersion today should
forecast volatility tomorrow. The return-prediction literature uses
dispersion the other way (Diether, Malloy, and Scherbina 2002: high
dispersion predicts *lower* subsequent returns); using dispersion as a
*volatility* predictor is the unpublished twist. The VOL campaign died
NO-GO against GARCH(1,1) (`docs/VOL_NEGATIVE_RESULT.md`) and the
HARVIX rematch died NO-GO (`docs/HARVIX_NEGATIVE_RESULT.md`). This is
a NEW, SEPARATELY pre-registered campaign on the same frozen target.
Nothing about VOL's or HARVIX's results carries a gate claim into
this campaign except the frozen target/universe/frame definitions,
which are identical by design.

The test period (2026-07-01..2026-08-31) and the confirmation period
(t0 >= 2026-09-01) remain untouched until the pre-registered
evaluation point described below.

## Frozen target

- **Target:** `realized_vol_5d`, EXACTLY the same definition as VOL:
  sample std (ddof=1) of daily log returns over the 5 trading days
  (t0, t0+5], decimal, daily (`docs/TARGETS_AND_FEATURES.md` v2.0).
- **Frame:** one row per (ticker, trading day), ALL ticker-days (not
  event-restricted). Rows whose 5-day window is partial are DROPPED,
  never filled. Features use only information knowable at t0's close.
  The label window (t0, t0+5] never enters a feature.

## Frozen periods

- **Development:** t0 <= 2026-06-30. All fitting and walk-forward
  happen here.
- **Test:** 2026-07-01..2026-08-31. Untouched during development. The
  single authorized test evaluation runs only after a dev GO (see
  decision rule).
- **Confirmation:** t0 >= 2026-09-01. NEVER read, scored, or
  inspected, for any purpose including debugging.
  `periods.check_no_confirmation` remains the enforcement point on
  every frame build, split, and evaluation call.

## Universe and data (dev)

- **Universe:** the frozen `data/universe_vol.csv`: S&P 100 (101) +
  SPY = 102 tickers. No change from VOL/HARVIX.
- **Prices:** yfinance daily OHLCV, 2022-01-01 through 2026-07-15
  (same download as VOL; same quality gates; download gaps logged,
  never backfilled silently).
- **VIX:** ^VIX and ^VIX3M daily closes, same series and burn-in rules
  as HARVIX (cached locally, uncommitted; fail-loud download).
- **NO GDELT anywhere in this campaign.** All features are
  price-derived.

## Dispersion feature spec (frozen, all point-in-time)

Let t range over the union exchange calendar (sorted unique trading
dates in the price panel). For each day t, using split/dividend
adjusted closes:

- r_{i,t} = ln(c_{i,t} / c_{i,t-1}) is ticker i's daily log return.
- **disp(t)** = sample std (ddof=1) of r_{i,t} over the 101 S&P 100
  tickers (the SPY market leg is EXCLUDED: the bet is about the
  constituents' cross-section). Only tickers with a finite r_{i,t}
  enter the cross-section; if fewer than 50 tickers are valid on day
  t, disp(t) = NaN.

Per frame row with forecast origin t0 (t0 is itself a union-calendar
trading day by frame construction; the builder asserts this and fails
loud on mismatch):

- `disp_level` = disp(t0)
- `disp_5d_change` = disp(t0) - disp(t0 - 5 trading days)
- `disp_z60` = (disp(t0) - mean(disp over the trailing 60 trading days
  ending at t0)) / std (ddof=1) of disp over the same 60 days
- `disp_level_x_rv1d` = disp_level * rv_1d (the stock's own daily RV
  leg from the frozen HAR set)
- `disp_5d_change_x_rv1d` = disp_5d_change * rv_1d
- `disp_z60_x_rv1d` = disp_z60 * rv_1d

**Point-in-time contract:** every dispersion input uses closes on days
<= t0 only. disp(t) for t <= t0 is computable at t's close, so the
dispersion series built on the dev panel is point-in-time safe for
every row (verified by a truncation test: features for t0 <= cut are
identical whether computed on the full panel or the panel truncated
at cut, mirroring the VOL point-in-time test).

**Burn-in:** rows with fewer than 65 trading days of price history
ending at t0 (60 dispersion trailing days + 5 for the change + 1 for
the day-t return leg) get NaN dispersion features and are DROPPED
with a logged count, never filled. The frozen challenger column set
is 31 HAR + 3 VIX + 6 dispersion = 40 features.

## Arms

- **naive (sanity check):** forecast = trailing `realized_vol_5d`
  (persistence). Reported, no gate authority.
- **garch11 (SECONDARY BASELINE):** the same scale-corrected
  GARCH(1,1) implementation as VOL (MLE on trailing 252 trading days,
  analytic 5-day term structure averaged to the target's daily scale;
  fit failures fall back to naive persistence and are counted, never
  hidden). Code is REUSED (`src/signal_lab/vol/garch.py`), not
  reimplemented. This is the Hansen-Lunde null, reported as the
  secondary comparison.
- **har_vix (PRIMARY BASELINE):** ridge regression (alpha = 1.0,
  StandardScaler fit on train only) on the frozen 34 HAR+VIX features
  (the HARVIX challenger definition, unchanged). The claim under test
  is narrowly "dispersion adds to the best history-based model," so
  HAR+VIX is the bar.
- **har_vix_disp (CHALLENGER):** the same frozen ridge on the 34
  HAR+VIX features plus the 6 frozen dispersion features above
  (40 features total).
- Forecasts from both ridge arms are floored at 1e-4 (same interface
  guard as VOL/HARVIX; never binding on real data).

## Selection honesty

- The estimator is frozen (ridge, alpha = 1.0, standardized): this is
  a one-arm feature test, not a model search. No selection split is
  run; the feature set is frozen by this document.
- The Deflated Sharpe Ratio (extraction item 6) is reported as a
  model-selection honesty metric: per-row QLIKE gains of the two
  tried ridge arms over naive as pseudo-returns, freq=1, pooled dev
  test rows. It does not gate the campaign.

## Validation

- 5-fold expanding purged walk-forward on t0, seed 7, min_train 50.
- **Purge:** drop train rows whose [t0, t0+5] label window overlaps
  the current fold's test range (j == k only, per the documented
  `make_splits` convention).
- **Embargo:** 5 trading days after each test fold (>= horizon).
- The splitter asserts its invariants on every call (no train/test
  overlap, no label-window overlap among kept rows, embargo zones
  empty).

## Primary metric

- **QLIKE** (Patton 2011): QLIKE(sigma^2, sigma_hat^2) =
  sigma^2/sigma_hat^2 - ln(sigma^2/sigma_hat^2) - 1. Lower is better.
- Secondary (reported, no gate authority): MSE on variance, MAE on
  volatility, Diebold-Mariano on the QLIKE loss differential vs the
  HAR+VIX baseline and vs GARCH(1,1), 95% Model Confidence Set.

## Decision rule (frozen)

- **Primary comparison:** har_vix_disp vs har_vix, paired per dev
  walk-forward fold on QLIKE.
- **Statistic:** mean paired QLIKE differential
  d = QLIKE(har_vix) - QLIKE(har_vix_disp) across folds; positive
  means the challenger beats HAR+VIX.
- **GO** iff mean(d) > 0 AND the lower bound of the 95% two-sided
  Student-t CI on d (df = n_folds - 1) is > 0. Otherwise NO-GO. A
  narrow miss is NO-GO; moving the bar after seeing results is
  forbidden.
- On GO: run the single authorized test-period evaluation
  (`confirm_test_eval=True`), report dev and test QLIKE side by side,
  then stop. Further work needs a new pre-registration.
- On NO-GO: stop, write `docs/DISPVOL_NEGATIVE_RESULT.md`. No peeking
  at test or confirmation, no target switching inside this campaign.

## Kill criterion (frozen)

- If the primary comparison is NO-GO, the campaign is DEAD. It is not
  revived by trying other models, other features, other horizons, or
  other metrics. Any of those is a new campaign with its own
  pre-registration.
- The pre-registered next bets are ATTRVOL and CONFMAG
  (`docs/SYNTHESIS_BETS.md`), each SEPARATELY pre-registered. Nothing
  about DISPVOL's data, features, or tuning carries a gate claim into
  them.

## Deviation policy

Any deviation from this document after its commit (target, features,
arms, horizon, metric, universe, periods, rule) is logged in
`docs/DISPVOL_RESULTS.md` (or `docs/DISPVOL_NEGATIVE_RESULT.md` on
NO-GO) under "Deviations", with rationale. Undocumented deviations
invalidate the pre-registration.

## Why it might fail (stated upfront, from the bet memo)

Dispersion may be a coincident rather than leading indicator (it
spikes *during* volatile regimes, which HAR already captures); the
cross-section of 100 large caps may carry too little idiosyncratic
disagreement to matter; any edge may live at horizons shorter than
5 days.
