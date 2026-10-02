# ATTRVOL Pre-registration: attribution structure predicts post-event volatility

**Status:** PRE-REGISTERED. This document is committed BEFORE any
ATTRVOL experiment is run. Any change to the frozen target, event
definition, features, arms, horizon, metric, or periods after this
commit is logged as a deviation in the result document.

**Context:** This is BET 2 (SPECULATIVE) from
`docs/SYNTHESIS_BETS.md`: *how* a move happened predicts what
volatility does next. An event-day return concentrated in one
idiosyncratic driver (single-firm news) resolves uncertainty fast, so
volatility should decay quickly (the Ederington-Lee 1996 event-vol
cycle). An event-day return that is diffuse across drivers, or that
resembles past noisy episodes, should see volatility persist. Two
price-computable proxies: (1) driver concentration from a market /
idiosyncratic decomposition of the event-day return; (2)
analogue-implied volatility from historical price-path analogues.
The VOL, HARVIX, EVENTVOL, TAILQ, and DISPVOL campaigns are all dead
(NO-GO). This is a NEW, SEPARATELY pre-registered campaign. Nothing
about their results carries a gate claim into this campaign except
the frozen target/universe/frame definitions, which are identical by
design.

The test period (2026-07-01..2026-08-31) and the confirmation period
(t0 >= 2026-09-01) remain untouched until the pre-registered
evaluation point described below.

## Frozen target

- **Target:** `realized_vol_5d`, EXACTLY the VOL definition: sample
  std (ddof=1) of daily log returns over the 5 trading days (t0, t0+5],
  decimal, daily (`docs/TARGETS_AND_FEATURES.md` v2.0).
- **Frame:** one row per (ticker, trading day), restricted to EVENT
  rows (see event definition). Rows whose 5-day window is partial are
  DROPPED, never filled. Features use only information knowable at
  t0's close. The label window (t0, t0+5] never enters a feature.

## Event definition (frozen, point-in-time)

- abn_ret(t0) = r_i(t0) - r_SPY(t0), the day's log return minus the
  SPY log return (simple market adjustment; computed from split/
  dividend-adjusted closes).
- Let Q90_i(t0) be the 90th percentile of |abn_ret| over ticker i's
  trailing 252 trading days ending at t0 (up to 252 days; fewer if
  history is short).
- Row (i, t0) is an EVENT row iff |abn_ret_i(t0)| >= Q90_i(t0) AND at
  least 60 valid trailing |abn_ret| observations exist. Otherwise it
  is not evaluated (it is not a "non-event control"; it is simply out
  of scope).
- The quantile is recomputed per t0 from trailing data only: the
  event set is point-in-time by construction.

## Attribution feature spec (frozen, all point-in-time)

Per EVENT row (ticker i, origin t0):

**Driver decomposition (market vs idiosyncratic).** No sector data
exists in the frozen universe file, so the decomposition is two-leg
(this is a deliberate frozen simplification, documented here):

- beta_i(t0) = OLS beta of ticker i on SPY over the trailing 252
  trading days ending at t0 (reuse `eventvol.features.trailing_beta`;
  fallback beta = 1.0 when fewer than 60 paired observations exist;
  fallbacks are counted, never hidden).
- market leg m = beta_i(t0) * r_SPY(t0); idiosyncratic leg
  e = r_i(t0) - m.
- denom = |m| + |e|; if denom == 0, the row's attribution features
  are NaN and the row is dropped (counted; cannot happen on a
  top-decile |abn_ret| day in practice, but the guard fails loud).
- s_m = |m| / denom; s_e = |e| / denom.
- `driver_H` = s_m^2 + s_e^2 in [0.5, 1.0]; 1.0 means the day's move
  is fully concentrated in one leg.
- `idio_share` = s_e (share of the move that is idiosyncratic).
- `ret_x_H` = r_i(t0) * driver_H (signed event return interacted
  with concentration).

**Price-path analogues (same ticker, trailing history only).**

- Candidate analogue days for event (i, t0): ticker i's trading days
  s with t0 - 500 <= s <= t0 - 5 (the s <= t0 - 5 bound guarantees
  the analogue's own 5-day label window ends at or before t0, so its
  realized_vol_5d label is knowable at t0).
- Path vector for day d: the 20 trailing daily log returns ending at
  d, z-scored (zero mean, unit sample std); candidates with zero path
  std are skipped.
- Similarity: cosine similarity between the event's z-scored path
  and each candidate's z-scored path.
- Take the k = 10 most similar candidates (fewer if fewer valid
  candidates exist; if fewer than 3, the row's analogue features are
  NaN and the row is dropped, counted).
- `analogue_vol_mean` = mean of the k analogues' realized_vol_5d.
- `analogue_vol_std` = sample std (ddof=1) of the k analogues'
  realized_vol_5d (0.0 if k == 1, but k >= 3 by the guard above).

**Frozen attribution feature set (5 features):**
`driver_H`, `idio_share`, `ret_x_H`, `analogue_vol_mean`,
`analogue_vol_std`.

**Point-in-time contract:** every input (returns, beta, quantile,
paths, analogue labels) uses closes on days <= t0 only, with the
analogue label bound s + 5 <= t0 enforced per candidate. Verified by
a truncation test mirroring the VOL/DISPVOL point-in-time tests.

## Arms

- **naive (sanity check):** forecast = trailing `realized_vol_5d`
  (persistence). Reported, no gate authority.
- **garch11 (SECONDARY BASELINE):** the same scale-corrected
  GARCH(1,1) implementation as VOL (MLE on trailing 252 trading days,
  analytic 5-day term structure averaged to the target's daily scale;
  fit failures fall back to naive persistence and are counted, never
  hidden). Code is REUSED (`src/signal_lab/vol/garch.py`), not
  reimplemented. Reported as the secondary comparison.
- **har_vix (PRIMARY BASELINE):** ridge regression (alpha = 1.0,
  StandardScaler fit on train only) on the frozen 34 HAR+VIX features
  (the HARVIX challenger definition, unchanged). The claim under test
  is narrowly "attribution structure adds to the best history-based
  model on the days when structure should matter most," so HAR+VIX is
  the bar.
- **har_vix_attr (CHALLENGER):** the same frozen ridge on the 34
  HAR+VIX features plus the 5 frozen attribution features above
  (39 features total).
- Forecasts from both ridge arms are floored at 1e-4 (same interface
  guard as VOL/HARVIX/DISPVOL; never binding on real data).

## Selection honesty

- The estimator is frozen (ridge, alpha = 1.0, standardized): this is
  a one-arm feature test, not a model search. No selection split is
  run; the feature set is frozen by this document.
- The Deflated Sharpe Ratio (extraction item 6) is reported as a
  model-selection honesty metric: per-row QLIKE gains of the two
  tried ridge arms over naive as pseudo-returns, freq=1, pooled dev
  event test rows. It does not gate the campaign.

## Validation

- 5-fold expanding purged walk-forward on t0, seed 7, min_train 50,
  over the EVENT-row frame (same splitter, same purge/embargo
  semantics as VOL: purge train rows whose [t0, t0+5] label window
  overlaps the fold's test range, j == k; 5-trading-day embargo).
- Fold-level statistics are computed over the EVENT rows in each
  fold's test set. Full-sample (all ticker-day) performance is
  reported as a secondary, with no gate authority.

## Primary metric

- **QLIKE** (Patton 2011): QLIKE(sigma^2, sigma_hat^2) =
  sigma^2/sigma_hat^2 - ln(sigma^2/sigma_hat^2) - 1. Lower is better.
- Secondary (reported, no gate authority): MSE on variance, MAE on
  volatility, Diebold-Mariano on the QLIKE loss differential vs the
  HAR+VIX baseline and vs GARCH(1,1), 95% Model Confidence Set.

## Decision rule (frozen)

- **Primary comparison:** har_vix_attr vs har_vix, paired per dev
  walk-forward fold on QLIKE over event-day test rows.
- **Statistic:** mean paired QLIKE differential
  d = QLIKE(har_vix) - QLIKE(har_vix_attr) across folds; positive
  means the challenger beats HAR+VIX on event days.
- **GO** iff mean(d) > 0 AND the lower bound of the 95% two-sided
  Student-t CI on d (df = n_folds - 1) is > 0. Otherwise NO-GO. A
  narrow miss is NO-GO; moving the bar after seeing results is
  forbidden.
- On GO: run the single authorized test-period evaluation
  (`confirm_test_eval=True`), report dev and test QLIKE side by side,
  then stop. Further work needs a new pre-registration.
- On NO-GO: stop, write `docs/ATTRVOL_NEGATIVE_RESULT.md`. No peeking
  at test or confirmation, no target switching inside this campaign.

## Kill criterion (frozen)

- If the primary comparison is NO-GO, the campaign is DEAD. It is not
  revived by trying other models, other features, other horizons, or
  other metrics. Any of those is a new campaign with its own
  pre-registration.
- The pre-registered next bet is CONFMAG (`docs/SYNTHESIS_BETS.md`),
  SEPARATELY pre-registered. Nothing about ATTRVOL's data, features,
  or tuning carries a gate claim into it.

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
  SPY = 102 tickers. No change from VOL/HARVIX/DISPVOL.
- **Prices:** yfinance daily OHLCV, 2022-01-01 through 2026-07-15
  (same download as VOL; same quality gates; download gaps logged,
  never backfilled silently).
- **VIX:** ^VIX and ^VIX3M daily closes, same series and burn-in rules
  as HARVIX (cached locally, uncommitted; fail-loud download).
- **NO GDELT anywhere in this campaign.** All features are
  price-derived.

## Deviation policy

Any deviation from this document after its commit (target, event
definition, features, arms, horizon, metric, universe, periods, rule)
is logged in `docs/ATTRVOL_RESULTS.md` (or
`docs/ATTRVOL_NEGATIVE_RESULT.md` on NO-GO) under "Deviations", with
rationale. Undocumented deviations invalidate the pre-registration.

## Why it might fail (stated upfront, from the bet memo)

The two-leg CAPM decomposition may be too coarse to capture true
driver structure; analogues by price path may match noise, not
mechanism; event-day vol may already be saturated by the event-day
return magnitude itself (which HAR includes via rv_1d), leaving
nothing for structure to add.
