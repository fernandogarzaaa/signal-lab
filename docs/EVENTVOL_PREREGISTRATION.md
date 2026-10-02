# EVENTVOL Pre-registration: event-window realized volatility vs implied volatility

**Status:** PRE-REGISTERED. This document is committed BEFORE any
event-volatility experiment is run. Any change to the frozen target,
event definition, baselines, arms, metric, validation, decision rule, or
periods after this commit is logged as a deviation in
`docs/EVENTVOL_RESULTS.md` (or `docs/EVENTVOL_NEGATIVE_RESULT.md` on
NO-GO). Undocumented deviations invalidate the pre-registration.

**Context:** Two campaigns are dead (direction: docs/NEGATIVE_RESULT.md;
volatility: docs/VOL_NEGATIVE_RESULT.md). The research sprint
(docs/RESEARCH_SPRINT.md, section 1.4 and Candidate A) identifies the
highest-conviction next question: does anything add to implied
volatility around earnings events? This is a NEW, SEPARATELY
pre-registered campaign on a different target and frame (earnings-event
rows, not ticker-days). It is not a continuation of the VOL campaign and
does not reuse its conclusions, although it reuses its frozen
definitions (realized_vol_5d v2.0), its validation machinery, and its
scale-corrected GARCH baseline. The test period (2026-07-01..2026-08-31)
and the confirmation period (t0 >= 2026-09-01) remain untouched until the
pre-registered evaluation point described below.

## STEP 0: IV baseline feasibility resolution (binding)

docs/SYNTHESIS_BETS.md flags that historical per-stock implied
volatility is not available from yfinance and forbids committing a
pre-registration against an unobtainable baseline. Resolved 2026-10-02
with live probes from this machine:

- yfinance option chains are current-only. AAPL on 2026-10-02 shows 22
  expiries, all dated 2026-10-02..2029-01-19: no historical per-stock IV
  exists to download. Option (b) (find a free historical per-stock IV
  source) is rejected: none was found that is free, historical, and
  verifiable from this machine.
- Pure option (a) (restrict the IV baseline to SPY/index) does not fit a
  per-stock earnings design: SPY has no earnings events.
- **Adopted: option (c), deliberate reformulation.** The IV baseline is
  the market's forward-looking volatility forecast proxied by the index:
  `iv_proxy` = beta(i, t0) x VIX(t0) / 100 / sqrt(252), decimal daily.
  VIX history (^VIX) and the 3-month term leg (^VIX3M) were verified
  downloadable from yfinance on 2026-10-02 (21 trading days for Jan 2024
  returned; closes 13.31..14.35 for ^VIX, 15.14..15.96 for ^VIX3M).
  beta(i, t0) is the OLS slope of the stock's daily log returns on SPY's
  over the trailing 252 trading days ending at t0 (minimum 60
  observations; fewer falls back to beta = 1.0 and is counted).

Why this is the right bar: (i) it is the market's own price of future
volatility (S&P 500 option prices), not a fitted history model; the
campaign question is whether anything adds to what the market already
prices, and only an option-implied quantity can stand for that;
(ii) the literature documents that implied volatility subsumes
history-based models (Blair, Poon, and Taylor 2001, on S&P 100
volatility at all horizons 1-20 days), so IV is the harshest honest
baseline, harsher than GARCH; (iii) the beta translation is the
standard market-model mapping from index vol to stock vol and uses only
information knowable at t0. It is labeled a proxy everywhere it
appears. The campaign does not claim to test per-stock ATM IV itself;
it tests whether price-derived event features add to the best
obtainable market-implied forecast.

## Frozen target

- **Target:** `realized_vol_5d`, versioned in `TargetConfig` target
  version 2.0 (docs/TARGETS_AND_FEATURES.md): sample std (ddof=1) of daily
  log returns over the 5 trading days (t0, t0+5], decimal, daily.
- **Frame:** one row per earnings event (ticker, t0). Rows whose 5-day
  window is partial (fewer than 5 forward trading days with returns)
  have NaN targets and are DROPPED, never filled. This is a regression
  task on event rows, not the ticker-day frame of the VOL campaign.
- **Forecast origin:** features use only information knowable at t0's
  close (returns through t0's close, prices through t0's close, VIX
  through t0's close). The label window (t0, t0+5] never enters a
  feature. The current quarter's reported EPS/surprise is NOT a feature
  (not knowable at t0's close for after-close announcements); only the
  previous quarter's surprise is used (see event features).

## Event definition (point-in-time rules, frozen)

- **Source:** yfinance `get_earnings_dates` per universe ticker, cached
  locally (not committed). Verified reachable 2026-10-02: AAPL returns
  rows back to 2020 with timestamps (e.g. 2026-10-29 16:00:00-04:00),
  EPS Estimate, Reported EPS, and Surprise(%).
- **Actuals only:** keep rows with non-NaN Reported EPS. Rows with only
  an EPS estimate (future scheduled announcements) are excluded; they
  are not events yet.
- **t0 mapping:** the calendar date of the earnings timestamp. If that
  date is a trading day, t0 is that date; otherwise t0 is the next
  trading day (weekend/holiday announcements). After-close (16:00 ET)
  and pre-open announcements both map to the timestamp's calendar date;
  in both cases t0's close is knowable at the forecast origin, and the
  label window (t0, t0+5] excludes t0's own return.
- **Dedup:** one row per (ticker, t0); if multiple reported rows map to
  the same t0, keep the latest timestamp.
- **Point-in-time caveat (documented assumption):** yfinance exposes no
  revision history for earnings dates, so the design uses the reported
  date as the event date, i.e. it assumes the date the market observed
  equals the reported date. Estimated-future rows are excluded, so no
  forward-looking schedule information enters.
- `periods.check_no_confirmation` is enforced on t0 on EVERY frame
  build: confirmation rows can never enter a development frame.

## Frozen periods

- **Development:** t0 <= 2026-06-30. All fitting and walk-forward happen
  here.
- **Test:** 2026-07-01..2026-08-31. Untouched during development. The
  single authorized test evaluation runs only after the dev campaign
  reaches GO (see decision rule).
- **Confirmation:** t0 >= 2026-09-01. NEVER read, scored, or inspected,
  for any purpose including debugging. `periods.check_no_confirmation`
  remains the enforcement point.

## Universe and prices (dev)

- **Universe:** S&P 100 constituents as of the pre-campaign date, plus
  SPY as the market leg. Exact ticker list is the FROZEN
  `data/universe_vol.csv` from the VOL campaign (101 + SPY = 102
  tickers); SPY has no earnings events and appears in the event frame
  only via the market leg (beta estimation, SPY feature legs).
- **Prices:** yfinance daily OHLCV, 2022-01-01 through 2026-07-15 (same
  window as VOL: the 2022 start gives the trailing 252-day GARCH and
  beta estimation windows a full burn-in before the first dev events).
  Download gaps are logged explicitly and never backfilled silently.
- **VIX:** yfinance ^VIX and ^VIX3M daily closes, 2022-01-01 through
  2026-07-15, cached locally. Missing VIX closes on event t0 use the
  most recent prior trading day's close and are counted.
- **Data-quality gates** (fail loudly): duplicate (ticker, date) rows,
  timestamp ordering violations, missing closes for labeled rows,
  zero-volume stretches inconsistent with the exchange calendar.

## Baselines and arms

- **naive (sanity check):** forecast = trailing realized_vol_5d
  (sample std, ddof=1, of daily log returns over the 5 trading days
  ending at t0). Reported, no gate authority.
- **iv_proxy (PRIMARY BASELINE):** beta(i, t0) x VIX(t0) / 100 /
  sqrt(252), decimal daily. beta from trailing 252 trading days of
  daily log returns through t0 (min 60 obs; fewer falls back to 1.0,
  counted). VIX missing at t0 falls back to the prior trading day's
  close, counted. Forecast floored at 1e-6. This is the market's
  forward-looking forecast (see STEP 0); the primary comparison is
  against it.
- **garch11 (secondary baseline):** GARCH(1,1) fit by maximum likelihood
  on the trailing 252 trading days of daily log returns through t0.
  Forecast = the scale-corrected 5-day term structure from the VOL
  campaign: sqrt of the MEAN of the 5 h-step variances (the VOL
  deviation ruling is applied from the start here, not re-derived).
  Fit failures (non-convergence, alpha+beta >= 1) fall back to the
  naive forecast and are counted, never hidden.
- **har_vix (secondary baseline):** ridge regression (alpha=1.0,
  standardized) on the frozen VOL HAR feature set (31 columns:
  rv_1d/5d/22d, sqret_lag1..5, parkinson_1d/5d, logvol, dlogvol_1d/5d,
  own + SPY legs, dow_0..4) PLUS vix features: vix_level
  (VIX/100/sqrt(252)), vix_5d_change, vix_runup_22d (VIX(t0) minus
  trailing-22d mean), vix_slope (VIX3M/VIX - 1). Hyperparameters frozen;
  no selection.
- **challenger (PRIMARY ARM):** LightGBM on the har_vix feature set PLUS
  event features (Ederington-Lee 1996 IV creation/resolution cycle
  proxies, all knowable at t0's close):
  - event_day_absret = |r(t0)|, event_day_ret = r(t0)
  - trailing_surprise = previous quarter's Surprise(%)/100 (the current
    quarter's surprise is excluded as not knowable at t0's close)
  - beta_252 (the same trailing beta used by iv_proxy)
  LightGBM hyperparameters are the frozen VOL set, pre-declared before
  any EVENTVOL run: n_estimators=300, lr=0.05, leaves=31, min_child=100,
  feat_frac=0.8, bag_frac=0.8, l2=1.0, seed=7. No hyperparameter or
  feature selection occurs in this campaign (the only selection in the
  program's history, ridge vs LightGBM, was VOL's; LightGBM is adopted
  as the challenger by design, documented here).

## Selection honesty

No model or feature selection runs in this campaign: both ML arms use
frozen hyperparameters and frozen feature lists declared above. The
Deflated Sharpe Ratio (extraction item 6) is reported as a
model-selection honesty metric on the walk-forward QLIKE gains over
naive. It does not gate the campaign.

## Validation

- 5-fold expanding purged walk-forward on t0, seed 7, min_train 50.
- **Purge:** drop train rows whose [t0, t0+5] label window overlaps the
  current fold's test range.
- **Embargo:** 5 trading days after each test fold (>= horizon).
- The splitter asserts its invariants on every call (no train/test
  overlap, no label-window overlap among kept rows, embargo zones
  empty). Reuses `signal_lab.validation.splits` unchanged.

## Primary metric

- **QLIKE** (Patton 2011): QLIKE(sigma^2, sigma_hat^2) =
  sigma^2/sigma_hat^2 - ln(sigma^2/sigma_hat^2) - 1. Lower is better.
  Consistent for ranking volatility forecasts and robust to noise in
  the realized proxy; gives the greatest power in Diebold-Mariano tests
  (Patton 2011).
- Secondary (reported, no gate authority): MSE on variance, MAE on
  volatility, Diebold-Mariano on the fold-level QLIKE loss differential
  vs the iv_proxy baseline.

## Model Confidence Set (second GO conjunct)

- **Set:** {challenger, iv_proxy, garch11, har_vix, naive} on per-row
  QLIKE losses pooled across the dev walk-forward test folds.
- **Procedure:** Hansen, Lunde, and Nason (2011) MCS with the range
  statistic T_R = max |t_ij| over pairwise loss differentials,
  alpha = 0.05. Bootstrap: day-block stationary bootstrap (mean block
  length 10 trading days, B = 5000 replications) over the pooled test
  rows, resampling whole trading days to preserve the cross-section.
  Models are eliminated iteratively until no rejection; the survivor
  set is the 95% MCS.
- **Documented limitation:** with 5 walk-forward folds the pooled rows
  carry within-fold dependence the day-block bootstrap only partly
  captures; the MCS is reported with this caveat and is used as a
  conjunct, not as the sole gate.

## Decision rule (frozen)

- **Primary comparison:** challenger vs iv_proxy, paired per dev
  walk-forward fold on QLIKE: d_k = QLIKE(iv_proxy) - QLIKE(challenger),
  positive favors the challenger.
- **Statistic:** mean paired QLIKE differential d across folds; 95%
  two-sided Student-t CI on d (df = n_folds - 1).
- **GO** iff ALL of: mean(d) > 0 AND the CI lower bound > 0 AND the
  challenger survives the 95% MCS. Otherwise NO-GO. A narrow miss on any
  conjunct is NO-GO; moving the bar after seeing results is forbidden.
- On GO: run the single authorized test-period evaluation
  (train on full dev, score 2026-07-01..2026-08-31 once), report dev and
  test QLIKE side by side in docs/EVENTVOL_RESULTS.md, then STOP.
  Further work needs a new pre-registration.
- On NO-GO: stop, write docs/EVENTVOL_NEGATIVE_RESULT.md with the same
  rigor as docs/VOL_NEGATIVE_RESULT.md (per-fold table, verdict
  computation step by step, deviations, kill-criterion acknowledgment).
  No peeking at test or confirmation, no target switching inside this
  campaign.

## Kill criterion (frozen)

- If the campaign is NO-GO on any GO conjunct, the campaign is DEAD. It
  is not revived by trying other models, other features, other
  horizons, other metrics, or other event definitions. Any of those is
  a new campaign with its own pre-registration.
- Nothing about EVENTVOL's data, features, or tuning carries a gate
  claim into any follow-up campaign.

## Deviation policy

Any deviation from this document after its commit (target, event
definition, baselines, arms, metric, validation, MCS procedure,
universe, periods, rule) is logged in docs/EVENTVOL_RESULTS.md (or
docs/EVENTVOL_NEGATIVE_RESULT.md on NO-GO) under "Deviations", with
rationale. Undocumented deviations invalidate the pre-registration.
