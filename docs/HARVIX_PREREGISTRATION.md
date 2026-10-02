# HARVIX Pre-registration: HAR+VIX rematch on realized_vol_5d

**Status:** PRE-REGISTERED. This document is committed BEFORE any
HARVIX experiment is run. Any change to the frozen target, baseline,
arms, features, horizon, metric, or periods after this commit is logged
as a deviation in the result document.

**Context:** The VOL campaign died NO-GO against GARCH(1,1) on QLIKE
(mean paired diff +0.0115, 95% CI straddling zero;
docs/VOL_NEGATIVE_RESULT.md). The research memo
(docs/RESEARCH_SPRINT.md, section 4 Candidate B, section 1.4) identifies
the untested literature arm: Blair, Poon, and Taylor (2001) show VIX
gives the most accurate out-of-sample S&P 100 volatility forecasts at
all horizons 1-20 days and subsumes history-based models; Pong,
Shackleton, Taylor, and Xu (2004) show IV-plus-model combinations beat
either alone. The VOL NO-GO was against GARCH without implied vol; the
literature's highest-conviction arm was never tested. This is Candidate
B: a one-arm rematch, not a fishing expedition. It is a SEPARATELY
pre-registered campaign on the same target. Nothing about VOL's data,
features, or tuning carries a gate claim into it, except the frozen
target/universe/frame definitions, which are identical by design.

The test period (2026-07-01..2026-08-31) and the confirmation period
(t0 >= 2026-09-01) remain untouched until the pre-registered evaluation
point described below.

## Frozen target

- **Target:** `realized_vol_5d`, EXACTLY the same definition as VOL:
  sample std (ddof=1) of daily log returns over the 5 trading days
  (t0, t0+5], decimal, daily (docs/TARGETS_AND_FEATURES.md v2.0).
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
- **Confirmation:** t0 >= 2026-09-01. NEVER read, scored, or inspected,
  for any purpose including debugging. `periods.check_no_confirmation`
  remains the enforcement point on every frame build, split, and
  evaluation call.

## Universe and data (dev)

- **Universe:** the frozen `data/universe_vol.csv`: S&P 100 (101) + SPY
  = 102 tickers. No change from VOL.
- **Prices:** yfinance daily OHLCV, 2022-01-01 through 2026-07-15 (same
  download as VOL; same quality gates; download gaps logged, never
  backfilled silently).
- **VIX:** ^VIX and ^VIX3M daily closes, same window. Availability
  check: ^VIX3M was verified reachable from yfinance on 2026-10-02 by
  the EVENTVOL campaign (downloaded to the local, uncommitted cache
  `data/eventvol_vix.parquet`). This campaign re-downloads both series
  at build time and FAILS LOUDLY if either download fails or returns
  zero rows: the campaign cannot run without the term-structure slope.

## Arms

- **naive (sanity check):** forecast = trailing `realized_vol_5d`
  (persistence). Reported, no gate authority.
- **garch11 (PRIMARY BASELINE):** the same scale-corrected GARCH(1,1)
  implementation as VOL (MLE on trailing 252 trading days, analytic
  5-day term structure averaged, i.e. sqrt of the MEAN of the h-step
  variances, to the target's daily scale; fit failures fall back to
  naive persistence and are counted, never hidden). Code is REUSED
  (`src/signal_lab/vol/garch.py`), not reimplemented.
- **har_vix (CHALLENGER):** ridge regression (alpha = 1.0,
  StandardScaler fit on train only) on the 31 HAR-style point-in-time
  price features (own leg + SPY legs + day-of-week, same
  `src/signal_lab/vol/features.py` FEATURE_COLUMNS as VOL) PLUS the 3
  frozen VIX features below (34 features total). Ridge is the frozen
  estimator for this arm: it is the literature's standard HAR-X
  estimation choice, and freezing it before the campaign keeps this a
  one-arm test rather than a model search.
- **har (SECONDARY BASELINE):** ridge regression (alpha = 1.0,
  standardized) on the 31 HAR features only. Isolates what VIX adds:
  the har_vix vs har comparison is reported with the same paired tests
  but has NO gate authority.
- Forecasts from both ridge arms are floored at 1e-4 (same interface
  guard as VOL; never binding on real data).

## VIX feature spec (frozen, all point-in-time)

All three features are computed from VIX trading-day closes on days
<= t0 only. t0 maps to the latest VIX trading day <= t0. Rows with
fewer than 22 VIX trading days of history ending at t0, or with
non-positive VIX/VIX3M closes at t0, get NaN VIX features and are
DROPPED with a logged count (never filled).

- `vix_level` = VIX(t0) / 100 / sqrt(252): decimal daily implied vol.
- `vix_5d_change` = VIX(t0) - VIX(t0 - 5 VIX trading days): VIX points.
- `vix_slope` = VIX3M(t0) / VIX(t0) - 1: term-structure slope
  (dimensionless; positive = upward-sloping term structure).

These are shared across tickers for a given t0 (a market-wide leg, as
the literature treats VIX for individual equities).

## Selection honesty

No model or hyperparameter selection runs in this campaign: all
estimators and feature lists are frozen above before any data is
touched. The Deflated Sharpe Ratio is reported as a model-selection
honesty metric (per-row QLIKE gains of har_vix over naive as
pseudo-returns, freq=1). It does not gate the campaign.

## Validation

- 5-fold expanding purged walk-forward on t0, seed 7, min_train 50.
- **Purge:** drop train rows whose [t0, t1] label window overlaps the
  current fold's test range (j == k only, per the documented
  `make_splits` deviation note).
- **Embargo:** 5 trading days after each test fold (>= horizon).
- The splitter asserts its invariants on every call.

## Primary metric

- **QLIKE** (Patton 2011) on variance (lower is better).
- Standard errors: HAC long-run variance form of the Diebold-Mariano
  statistic. Documented equivalence
  (src/signal_lab/vol/metrics.py): with one observation per fold on
  non-overlapping blocks, the HAC long-run variance collapses to the
  sample variance of the fold means, so the reported paired-t form IS
  the DM statistic; both the statistic and the two-sided p-value are
  reported.
- **Model Confidence Set** (Hansen, Lunde, Nason 2011): 95% MCS via the
  range statistic T_R with day-block stationary bootstrap (B = 5000,
  mean block length 10 trading days, seed 7; code REUSED from
  src/signal_lab/eventvol/mcs.py) over {har_vix, garch11, har, naive}
  on pooled per-row dev QLIKE. REPORTED, not a gate: the frozen GO
  rule below is VOL's rule, unchanged.
- Secondary (reported, no gate authority): MSE on variance, MAE on vol.

## Decision rule (frozen)

- **Primary comparison:** har_vix vs garch11, paired per dev
  walk-forward fold on QLIKE.
- **Statistic:** mean paired QLIKE differential d = QLIKE(garch11) -
  QLIKE(har_vix) across folds; positive means HAR+VIX beats GARCH.
- **GO** iff mean(d) > 0 AND the lower bound of the 95% two-sided
  Student-t CI on d (df = n_folds - 1) is > 0. Otherwise NO-GO. A
  narrow miss is NO-GO; moving the bar after seeing results is
  forbidden.
- On GO: run the single authorized test-period evaluation
  (`confirm-test-eval`), report dev and test QLIKE side by side, write
  docs/HARVIX_RESULTS.md, then STOP.
- On NO-GO: stop, write docs/HARVIX_NEGATIVE_RESULT.md. No peeking at
  test or confirmation, no target switching inside this campaign.

## Kill criterion (frozen)

- If the primary comparison is NO-GO, the campaign is DEAD. It is not
  revived by trying other models, other features, other horizons, or
  other metrics. Any of those is a new campaign with its own
  pre-registration.

## Deviation policy

Any deviation from this document after its commit (target, baseline,
arms, features, horizon, metric, universe, periods, rule) is logged in
the result document under "Deviations", with rationale. Undocumented
deviations invalidate the pre-registration.
