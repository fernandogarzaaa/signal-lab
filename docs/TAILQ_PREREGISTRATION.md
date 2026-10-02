# TAILQ Pre-registration: tail-quantile (VaR) forecasting around earnings events

**Status:** PRE-REGISTERED. This document is committed BEFORE any
tail-quantile experiment is run. Any change to the frozen event
definition, target, arms, CAViaR specification, metric, validation,
decision rule, or periods after this commit is logged as a deviation in
`docs/TAILQ_RESULTS.md` (or `docs/TAILQ_NEGATIVE_RESULT.md` on NO-GO).
Undocumented deviations invalidate the pre-registration.

**Context:** Three campaigns are dead (direction:
docs/NEGATIVE_RESULT.md; volatility: docs/VOL_NEGATIVE_RESULT.md;
HAR+VIX rematch: docs/HARVIX_NEGATIVE_RESULT.md; event volatility:
docs/EVENTVOL_NEGATIVE_RESULT.md). The research sprint
(docs/RESEARCH_SPRINT.md, section 2.4 and Candidate C) identifies the
next literature-backed question: tail-quantile (Value at Risk)
forecasting around events, with Engle-Manganelli CAViaR (2004, JBES) as
the canonical baseline. This is a NEW, SEPARATELY pre-registered
campaign on a different target (a conditional quantile, not a mean or
variance). It is not a continuation of any prior campaign and does not
reuse their conclusions, although it reuses frozen infrastructure: the
earnings event pipeline (`src/signal_lab/eventvol/earnings.py`), the
purged walk-forward splitter (`src/signal_lab/validation/splits.py`),
the GARCH implementation (`src/signal_lab/vol/garch.py`), the MCS
(`src/signal_lab/eventvol/mcs.py`), and the frozen periods
(`src/signal_lab/validation/periods.py`). The test period
(2026-07-01..2026-08-31) and the confirmation period (t0 >= 2026-09-01)
remain untouched until the pre-registered evaluation point described
below.

## Frozen event definition (point-in-time, decided FIRST)

- **Event set: earnings announcements.** One row per (ticker, t0).
  Source: yfinance `get_earnings_dates` per universe ticker, via the
  frozen EVENTVOL pipeline (`download_earnings`): actuals only
  (non-NaN Reported EPS; estimate-only future rows excluded), latest
  timestamp wins per (ticker, calendar date), previous-quarter
  surprise carried as a knowable-at-t0 feature.
- **t0 mapping:** the calendar date of the earnings timestamp; if that
  date is not a trading day, t0 is the next trading day
  (`_map_t0`, unchanged).
- **Justification for choosing earnings over trailing top-decile
  |abn_ret| days:** (i) earnings are scheduled and point-in-time
  verifiable, so the event set is exogenous to the realized outcome;
  selecting trailing top-decile abnormal-return days conditions the
  frame on the magnitude being forecast and complicates coverage
  interpretation; (ii) the earnings pipeline is already built, tested,
  and point-in-time audited (docs/EVENTVOL_PREREGISTRATION.md); (iii)
  the sprint's Candidate C explicitly pairs the tail target with
  events, and earnings are the canonical scheduled event in the
  Ederington-Lee (1996) event-volatility literature.
- **Dedup:** one row per (ticker, t0); latest timestamp wins.
  `periods.check_no_confirmation` is enforced on t0 on EVERY build.

## Frozen target

- **Target:** the 5% conditional quantile of the 3-day event return.
  The 3-day event return for row (ticker, t0) is
  r_3d = sum of daily log returns over the 3 trading days
  (t0, t0+3] (t0+1, t0+2, t0+3). Decimal. Rows whose 3-day window is
  partial (fewer than 3 forward trading days) are DROPPED, never
  filled.
- **Forecast origin:** features use only information knowable at t0
  (returns and prices through t0's close, earnings actuals published
  at or before t0). The label window (t0, t0+3] never enters a
  feature.
- **Forecast:** a VaR-style lower-tail quantile forecast q_hat
  (expected negative), evaluated against r_3d at tau = 0.05.

## Frozen universe and prices (dev)

- **Universe:** S&P 100 constituents as of the pre-campaign date, plus
  SPY as the market leg, frozen in `data/universe_vol.csv` (101
  tickers + SPY). SPY has no earnings events and contributes no event
  rows; it is the market leg for the beta feature only.
- **Prices:** yfinance daily OHLCV, 2022-01-01 through 2026-07-15 (the
  VOL download, reused; quality gates passed). The 2022 start gives
  the trailing 252-day CAViaR/GARCH estimation window and the 66-day
  HAR monthly leg a full burn-in before the first dev event.
- **VIX:** ^VIX + ^VIX3M daily closes (the HARVIX download; fail-loud
  if unavailable).
- **Data-quality gates** (fail loudly): duplicate (ticker, t0) rows,
  timestamp ordering violations, missing closes for labeled rows.

## Frozen arms

### challenger: lgbm_quantile (PRIMARY)

LightGBM quantile regression (`objective="quantile"`, `alpha=0.05`)
on 15 frozen point-in-time features, refit per walk-forward fold on
that fold's train rows.

Frozen feature list (all knowable at t0's close):

1. `rv_5d`: trailing sample std (ddof=1) of daily log returns over
   (t0-5, t0], decimal daily.
2. `rv_22d`: same over (t0-22, t0].
3. `rv_66d`: same over (t0-66, t0].
4. `ret_3d_pre`: sum of log returns over (t0-3, t0].
5. `abs_ret_3d_pre`: absolute value of (4).
6. `vix_level` = VIX(t0)/100/sqrt(252) (HARVIX frozen definition).
7. `vix_5d_change` = VIX(t0) - VIX(t0 - 5 VIX trading days).
8. `vix_slope` = VIX3M(t0)/VIX(t0) - 1.
9. `beta_252`: OLS slope of the stock's daily log returns on SPY's
   over the trailing 252 trading days ending at t0 (minimum 60
   observations; fewer falls back to 1.0 and is counted).
10. `surprise`: Surprise(%)/100 from the earnings calendar row.
11. `abs_surprise`: absolute value of (10).
12. `prev_surprise`: previous quarter's Surprise(%)/100
    (EVENTVOL pipeline; NaN for first events is filled with 0.0 and
    the fill is fixed, never data-dependent).
13. `dow`: day-of-week of t0 (0..4).
14. `spy_ret_3d_pre`: sum of SPY log returns over (t0-3, t0].
15. `spy_rv_22d`: SPY trailing 22-day sample std (ddof=1).

Frozen hyperparameters (declared before any data is touched; no
selection, no tuning): `n_estimators=300`, `learning_rate=0.05`,
`num_leaves=15`, `min_child_samples=40`, `feature_fraction=0.8`,
`bagging_fraction=0.8`, `bagging_freq=1`, `lambda_l2=1.0`,
`seed=7`, `deterministic=True`, `verbose=-1`.

Rows with NaN features (insufficient trailing history) are dropped
with a logged count, never filled (except the fixed 0.0 fill for
first-event prev_surprise above).

**Justification for the challenger:** CAViaR models quantile dynamics
from the asset's own return history only. The thesis under test is
that event-specific and market-state information (earnings surprise,
pre-event drift, VIX level and term structure, market beta) shifts
the event-return tail beyond what autoregressive quantile dynamics
capture. Quantile gradient boosting is the standard flexible
estimator for this; the frozen hyperparameters mirror the VOL
campaign's LightGBM config (adapted leaves/child size for the
smaller event frame) and are not tuned.

### caviar (PRIMARY BASELINE): Engle-Manganelli (2004) canonical specification

The symmetric absolute value (SAV) CAViaR, the paper's canonical
specification:

    f_t(beta) = b1 + b2 * f_{t-1} + b3 * |r_{t-1}|

Estimated per (ticker, t0) by minimizing the quantile check
function at tau = 0.05:

    sum_t rho_tau(r_t - f_t(beta)),  rho_tau(u) = u * (tau - 1[u < 0])

over the trailing 252 trading-day daily log returns ending at t0
(minimum 100 observations; fewer triggers the fallback below).

Frozen estimation protocol (deterministic):

- Initialize f_1 at the empirical 5% quantile of the 252-day window.
- Optimize with `scipy.optimize.differential_evolution`
  (seed=7, maxiter=100, tol=1e-7), bounds
  b1 in [3*q05, 0] (q05 = empirical 5% quantile of the window,
  negative), b2 in [0.0, 0.999] (stationarity guard), b3 in [0.0,
  2.0]; then polish with Nelder-Mead from the DE solution
  (maxiter=500). No other starts, no tuning.
- 1-day VaR forecast: f_{T+1} = b1 + b2*f_T + b3*|r_T|.
- **Horizon adaptation (explicit, frozen):** the canonical model is a
  1-day model. The 3-day VaR forecast is
  q_hat = f_{T+1} * sqrt(3), the i.i.d. variance-time scaling. This
  is an approximation, documented here rather than hidden: it applies
  identically to caviar, garch_hs, and naive, so the arm comparison
  on the 3-day target is apples-to-apples.
- Fit failures (non-convergence, non-finite forecasts, short window)
  fall back to the naive forecast (below) and are COUNTED, never
  hidden.

### garch_hs (SECONDARY BASELINE): GARCH-filtered historical simulation VaR

- GARCH(1,1) by MLE on the trailing 252 trading days through t0
  (reuse `src/signal_lab/vol/garch.py`, unchanged).
- In-sample conditional variances sigma_t^2 from the fitted
  recursion; standardized residuals z_t = r_t / sigma_t.
- q05 = empirical 5% quantile of z (numpy quantile, linear
  interpolation).
- 1-day VaR = sigma_{t0+1|t0} * q05; 3-day VaR = 1-day VaR * sqrt(3)
  (same frozen horizon scaling).
- Fit failures fall back to the naive forecast and are counted.

### naive (SANITY, no gate authority)

Unconditional empirical 5% quantile of the trailing 252 trading-day
log returns through t0, times sqrt(3). Reported on every fold.

## Selection honesty

No model or hyperparameter selection runs in this campaign: the
challenger's estimator, feature list, and hyperparameters, and both
baselines' specifications, are frozen above before any data is
touched. The Deflated Sharpe Ratio (extraction item 6) is reported
as a model-selection honesty metric (per-row pinball gains over
naive as pseudo-returns). It does not gate the campaign.

## Validation

- 5-fold expanding purged walk-forward on t0, seed 7, min_train 50,
  `WalkForwardConfig(horizon_days=3, embargo_days=3)` (embargo =
  horizon, per the config's invariant).
- **Purge:** drop train rows whose [t0, t1] label window overlaps the
  current fold's test range (`make_splits`, documented rule).
- **Embargo:** 3 trading days after each test fold.
- The splitter asserts its invariants on every call.

## Primary metric

- **Pinball (quantile) loss at tau = 0.05** (lower is better):
  L_tau(r, q) = (r - q) * (tau - 1[r < q]).
  Pinball is the strictly consistent scoring rule for quantiles
  (the analog of Patton 2011's QLIKE argument for variances) and is
  linear in the tail, so no single row can dominate a fold the way
  QLIKE rows did in HARVIX fold 4.
- Secondary (reported, no gate authority): Kupiec unconditional
  coverage, Christoffersen independence and conditional coverage
  tests, 95% Model Confidence Set over the arms, per-fold pinball,
  Diebold-Mariano on fold-level pinball loss differentials (paired-t
  form, the documented HAC-equivalent on non-overlapping fold
  blocks), DSR.

## Coverage tests (frozen)

On the pooled dev walk-forward test rows ordered by (t0, ticker),
with hit I_i = 1[r_3d_i < q_hat_i]:

- **Kupiec (1995) unconditional coverage:** LR_uc =
  -2 log[ (p0^x (1-p0)^{n-x}) / (p_hat^x (1-p_hat)^{n-x}) ],
  p0 = 0.05, x = sum I, p_hat = x/n; chi2(1). Pass iff p > 0.05.
- **Christoffersen (1998) independence:** on the hit sequence,
  count n00, n01, n10, n11; LR_ind =
  -2 log[ ((1-pi)^{n00+n10} pi^{n01+n11}) /
          ((1-pi0)^{n00} pi0^{n01} (1-pi1)^{n10} pi1^{n11}) ],
  pi = (n01+n11)/(n00+n01+n10+n11), pi0 = n01/(n00+n01),
  pi1 = n11/(n10+n11); chi2(1). Pass iff p > 0.05. Documented
  limitation: the pooled ordering mixes tickers, so same-day
  cross-sectional dependence is not modeled; the test is applied as
  specified regardless.
- **Conditional coverage:** LR_cc = LR_uc + LR_ind, chi2(2).
  Pass iff p > 0.05.

## Decision rule (frozen, conjunctive)

- **Primary comparison:** challenger (lgbm_quantile) vs caviar,
  paired per dev walk-forward fold on pinball loss.
- **Statistic:** mean paired pinball differential
  d = pinball(caviar) - pinball(challenger) across folds; positive
  means the challenger beats CAViaR.
- **GO** iff ALL of the following hold:
  1. mean(d) > 0 AND the lower bound of the 95% two-sided Student-t
     CI on d (df = n_folds - 1) is > 0;
  2. the challenger PASSES all three coverage tests (Kupiec,
     Christoffersen independence, conditional coverage; p > 0.05
     each);
  3. caviar FAILS at least one coverage test (p <= 0.05).
  Otherwise NO-GO. A narrow miss is NO-GO; moving the bar after
  seeing results is forbidden.
- On GO: run the single authorized test-period evaluation
  (`confirm_test_eval=True`) on 2026-07-01..2026-08-31, report dev
  and test pinball side by side, then stop. Note: prices end
  2026-07-15, so the labelable test set is sparse (events with full
  3-day windows only); the labelable test row count is reported.
- On NO-GO: stop, write `docs/TAILQ_NEGATIVE_RESULT.md`. No peeking
  at test or confirmation, no target switching inside this campaign.

## Kill criterion (frozen)

- If the primary comparison is NO-GO, the campaign is DEAD. It is
  not revived by trying other models, other features, other
  quantiles, other horizons, other event sets, or other metrics. Any
  of those is a new campaign with its own pre-registration.
- Miss on pinball kills the campaign even if coverage looks good;
  coverage is a conjunct, not a substitute.

## Deviation policy

Any deviation from this document after its commit (event definition,
target, arms, CAViaR specification, features, hyperparameters,
metric, coverage tests, universe, periods, rule) is logged in the
result doc under "Deviations", with rationale. Undocumented
deviations invalidate the pre-registration.

## Periods (frozen)

- Development: t0 <= 2026-06-30. All fitting and walk-forward here.
- Test: 2026-07-01..2026-08-31. GO-gated single evaluation only.
- Confirmation: t0 >= 2026-09-01. NEVER read, scored, or inspected,
  for any purpose including debugging.
  `periods.check_no_confirmation` enforced on every frame build.
