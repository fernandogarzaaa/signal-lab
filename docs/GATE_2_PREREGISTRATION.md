# Gate 2 Pre-registration: final gate retest

**Status:** PRE-REGISTERED. This document was committed BEFORE any gate-2
experiment was run. Any change to the frozen target, horizon, models, or
confirmation cutoff after this commit is logged as a deviation in
`docs/GATE_2.md`.

**Context:** Gate 1 (docs/GATE.md, PR #35) fired NO-GO on 693 dev rows:
text over price-only, mean paired PR-AUC diff +0.0035, 95% CI
[-0.0091, 0.0160], p=0.48. Phases 4-7 were not built. The user authorized
ONE final retest with wider data and a FinBERT text arm. This run is
EXPLORATORY (registry id EXP-027 and following); it does not touch the
test period (2026-07-01..2026-08-31) and NEVER touches the confirmation
period (t0 >= 2026-09-01).

## Decision rule (frozen)

- **Primary comparison:** variant A (price-only) vs variant C (price +
  text), paired per dev walk-forward fold.
- **Primary metric:** mean paired PR-AUC difference (C minus A) across
  folds, primary model `logreg_balanced`.
- **Primary test:** two-sided 95% Student-t confidence interval on the
  paired differences (df = n_folds - 1), identical to gate 1.
- **GO** iff the mean paired difference is > 0 AND the lower bound of the
  95% CI is > 0. **Otherwise NO-GO.** No other outcome counts. A narrow
  miss is still NO-GO; widening the bar after seeing results is forbidden.
- On NO-GO: stop, write `docs/NEGATIVE_RESULT.md`. On GO: stop anyway and
  report; Phases 4-7 require separate explicit approval.

## Text arms (declared before seeing results)

- **TF-IDF arm (existing):** 5000-feature 1-2gram sublinear TF-IDF + 8
  Loughran-McDonald lexicon features, vectorizer fit on train texts only,
  per fold. Re-run and reported.
- **FinBERT arm (PRIMARY):** frozen `ProsusAI/finbert` 768-dim per-article
  embeddings (CLS pooling), deterministic function of article text only,
  therefore point-in-time by construction. Standardized per fold
  (StandardScaler fit on train only). **No dimensionality reduction in the
  primary analysis.** Cache-only contract: a missing embedding fails the
  run loudly, never zero-filled.
- Variant C for the primary comparison uses the FinBERT arm
  (C_finbert = price + FinBERT). C_tfidf, B_tfidf, B_finbert are reported
  as secondary comparisons with no gate authority.

## Frozen target, horizon, models

- **Target (unchanged from gate 1):** legacy weak label, `LabelConfig`
  defaults: label = 1 iff the ticker's cumulative abnormal return
  (ticker minus SPY) over the 3 trading days after t0 lands at or above
  the per-fold train-only 90th percentile; attention filter enabled
  (min 2 articles, top-quartile rule, train-only quantiles per fold).
- **Horizon:** 3 trading days (label window); t1 = t0 + 3 trading days.
- **Embargo / purge:** identical to gate 1: purge train rows whose
  [t0, t1] overlaps the current fold's test range; 5-trading-day embargo
  after each test fold; expanding window, 5 folds, min_train 50,
  seed 7, dedupe weight policy.
- **Models (unchanged):** naive, historical_rate, logreg_plain,
  logreg_balanced (primary), lightgbm_balanced.
- **Price features (unchanged):** the 21 dense features from gate 1
  (7 ctx + 14 price/volume/market), as-of anchored, scaler fit on train
  only per fold.
- Calibration and abstention diagnostics are NOT re-run: they were
  non-gate diagnostics in gate 1, both failed to help (calibration
  collapsed to constant predictors), and the gate rule does not involve
  them.

## Frozen periods

- **Development:** t0 <= 2026-06-30. All widening, selection, and the
  gate retest happen here.
- **Test:** 2026-07-01..2026-08-31. Untouched in this run.
- **Confirmation:** t0 >= 2026-09-01. NEVER read, scored, or inspected,
  for any purpose including debugging. `periods.check_no_confirmation`
  remains the enforcement point.

## Data widening (dev only)

- **Universe rule:** S&P 100 constituents (100 liquid US tickers) plus
  any original-10 watchlist ticker not in the index. Exact list committed
  as data/universe_gate2.csv in the widening step; count reported.
- **News window:** 2023-07-01 through 2026-06-30 (3 years), GDELT per
  ticker per quarter, max 250 records per query, 30s politeness sleep,
  exponential backoff on failure, per-(ticker, quarter) checkpointing in
  a fetch_log table. Gaps logged explicitly, never backfilled silently.
- **Prices:** yfinance daily, same tickers, 2023-07-01 through
  2026-07-15 (forward buffer covers label windows for late-June rows).
- **Quarantine:** any fetched row with t0 > 2026-06-30 is excluded from
  the gate-2 frame by the existing dev-end filter; it is not deleted
  (it belongs to test/confirmation) and never enters training.
- **Data-quality gates** (fail loudly): duplicate urls, timestamp
  ordering violations, missing prices for labeled rows, missing labels,
  future-dated articles relative to fetch time.

## Power note (to be completed in docs/GATE_2.md)

Gate 1's CI width was 0.0251 on 693 rows. The widened frame is expected
to yield 10-40x rows; GATE_2.md will report the achieved CI width and
the minimum detectable effect at 80% power given the observed fold-level
variance. A NO-GO on wide data with a tight CI is stronger evidence of
no edge than gate 1's NO-GO.

## Deviation policy

Any deviation from this document after its commit (target, horizon,
models, cutoff, arms, rule) is logged in docs/GATE_2.md under
"Deviations", with rationale. Undocumented deviations invalidate the
pre-registration.
