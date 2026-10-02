# TAILQ campaign: NEGATIVE RESULT (NO-GO, campaign dead)

**Pre-registration:** `docs/TAILQ_PREREGISTRATION.md` (frozen, binding;
merged as PR #64, commit 7226bf9). This is Candidate C from
docs/RESEARCH_SPRINT.md section 4: tail-quantile (VaR) forecasting
around earnings events with the canonical Engle-Manganelli (2004)
CAViaR baseline. Target: the 5% conditional quantile of the 3-day
event return r_3d = sum of daily log returns over (t0, t0+3], one
row per earnings event (yfinance earnings calendar, actuals only,
t0 = earnings date mapped to the next trading day). Challenger:
lgbm_quantile = LightGBM quantile regression (objective="quantile",
alpha=0.05; n_estimators=300, lr=0.05, leaves=15, min_child=40,
feat_frac=0.8, bag_frac=0.8, l2=1.0, seed=7, deterministic) on the 15
frozen point-in-time features (HAR legs, pre-event returns, VIX
level/change/slope, beta_252, surprise, prev_surprise, dow, SPY legs),
refit per fold. Baselines: caviar = per-(ticker, t0) symmetric
absolute value CAViaR f_t = b1 + b2*f_{t-1} + b3*|r_{t-1}| fit by
quantile-check-function minimization (differential_evolution seed 7
+ Nelder-Mead polish) on trailing 252 trading days, 1-day forecast
scaled by sqrt(3) (frozen horizon adaptation); garch_hs =
GARCH(1,1)-filtered historical simulation VaR (same sqrt(3) scaling);
naive = trailing empirical 5% quantile x sqrt(3) (sanity, no gate).
Metric: pinball loss at tau=0.05 PRIMARY; Kupiec unconditional
coverage, Christoffersen independence and conditional coverage;
95% Model Confidence Set (reported, not a gate); DSR honesty metric
(reported, not a gate). Validation: 5-fold expanding purged
walk-forward on t0, seed 7, min_train 50, horizon 3 / embargo 3
trading days. Universe: S&P 100 (101) + SPY, frozen in
`data/universe_vol.csv`. Prices: yfinance 2022-01-01..2026-07-15
(reused VOL download; quality gates passed). VIX: ^VIX + ^VIX3M
(reused HARVIX download).

## Verdict: NO-GO

The frozen decision rule (conjunctive): GO iff (1) mean paired pinball
differential d = pinball(caviar) - pinball(challenger) across folds
satisfies mean(d) > 0 AND the lower bound of the 95% two-sided
Student-t CI (df = n_folds - 1) is > 0; AND (2) the challenger passes
all three coverage tests (p > 0.05); AND (3) CAViaR fails at least one
coverage test. Otherwise NO-GO.

### Per-fold pinball loss (tau = 0.05)

| fold | test range | n_train | n_test | purged | embargoed | pinball challenger | pinball caviar | pinball garch_hs | pinball naive | d = caviar - challenger |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-11-17..2023-08-03 | 274 | 284 | 5 | 0 | 0.005283 | 0.004911 | 0.005252 | 0.004856 | -0.000372 |
| 2 | 2023-08-07..2024-04-30 | 553 | 282 | 10 | 0 | 0.009101 | 0.008690 | 0.008816 | 0.008507 | -0.000410 |
| 3 | 2024-05-01..2025-01-29 | 829 | 273 | 12 | 4 | 0.010135 | 0.011819 | 0.011746 | 0.011896 | +0.001685 |
| 4 | 2025-01-30..2025-10-21 | 1088 | 284 | 15 | 15 | 0.005699 | 0.006339 | 0.006795 | 0.006143 | +0.000640 |
| 5 | 2025-10-22..2026-06-24 | 1361 | 273 | 13 | 28 | 0.009252 | 0.008961 | 0.009526 | 0.009231 | -0.000290 |

### Verdict computation (step by step)

- d_k = pinball(caviar) - pinball(challenger) per fold:
  -0.000372, -0.000410, +0.001685, +0.000640, -0.000290
- mean(d) = +0.000250
- se(d) = 0.000408 (sample std, ddof=1, divided by sqrt(5))
- df = 4; t(0.975, 4) = 2.7764
- 95% two-sided Student-t CI on mean(d):
  [+0.000250 - 2.7764 x 0.000408, +0.000250 + 2.7764 x 0.000408]
  = [-0.000882, +0.001382]
- mean(d) > 0 is TRUE (+0.000250), but the CI lower bound > 0 is
  FALSE (-0.000882). A narrow miss is NO-GO; the bar is not moved.
- Diebold-Mariano on fold-level pinball loss differentials (paired-t
  form): stat = +0.6138, two-sided p = 0.5726, n = 5. No significant
  difference.
- Conjunct (1) FAILS. The kill criterion (miss on pinball kills the
  campaign) fires here; the coverage conjuncts are reported below for
  completeness.
- **Verdict: NO-GO.**

The challenger beats CAViaR on 2 of 5 folds and loses on 3; the mean
differential is +0.00025 with a CI spanning zero. Neither the
flexible quantile model nor the canonical quantile-dynamics model
separates from the other on pinball. Note the naive arm is
competitive throughout (best pinball on folds 1 and 2), which is
consistent with the finding below that all arms misjudge the tail.

### Coverage tests (pooled dev test rows, 1,396 rows ordered by (t0, ticker))

| arm | Kupiec LR_uc (p) | Christoffersen ind. (p) | conditional (p) | violation rate |
| --- | --- | --- | --- | --- |
| lgbm_quantile | 192.76 (0.0000) FAIL | 7.99 (0.0047) FAIL | 200.75 (0.0000) FAIL | 14.90% |
| caviar | 147.01 (0.0000) FAIL | 16.03 (0.0001) FAIL | 163.04 (0.0000) FAIL | 13.47% |
| garch_hs | - | - | - | 13.25% |
| naive | - | - | - | 13.11% |

Expected violation rate at tau = 0.05: 5%. Every arm over-violates by
2.6x to 3x: the 3-day earnings event returns are far more tail-risky
than any of the forecasts admit. The challenger fails all three
coverage tests, so conjunct (2) FAILS independently of the pinball
verdict. CAViaR fails all three as well, so conjunct (3) holds, but
the rule is conjunctive and the campaign is already dead on
conjuncts (1) and (2).

The systematic under-coverage across all four arms (including the
directly-estimated LightGBM quantile) says the failure is in the
frame, not the estimator: daily-calibrated tail models, even
augmented with event features, do not stretch far enough into the
earnings-event tail. The sqrt(3) i.i.d. horizon scaling (applied
identically to caviar, garch_hs, and naive per the frozen spec)
understates event-window tail risk; the challenger, which predicts
the 3-day quantile directly, under-covers just as badly, so the
scaling is not the whole story either.

### Model Confidence Set (95%, reported, not a gate)

Hansen-Lunde-Nason T_R, day-block stationary bootstrap, B=5000, mean
block 10 trading days, seed 7, over {lgbm_quantile, caviar,
garch_hs, naive} on 1,396 pooled per-row pinball losses: eliminated
garch_hs (p=0.0432); final round p=0.6541, no further elimination.
Survivors: lgbm_quantile, caviar, naive. The MCS cannot separate the
challenger from CAViaR or from the naive unconditional quantile; the
primary rule decides, and it says NO-GO.

### Selection honesty

No model or hyperparameter selection ran in this campaign: the
challenger's estimator, feature list, and hyperparameters were frozen
in the pre-registration before any data was touched. DSR honesty
metric (reported, not a gate; per-row pinball gains over naive as
pseudo-returns, freq=1, over the three evaluated arms):

- lgbm_quantile: Sharpe +0.0259, DSR 0.1136, likely_false_discovery = True
- caviar: Sharpe -0.0048, DSR 0.0112, likely_false_discovery = True
- garch_hs: Sharpe -0.1019, DSR 0.0000, likely_false_discovery = True

## Fit failures and effective row counts

- CAViaR naive fallbacks on walk-forward test rows: 20 / 1,396
  (1.43%), counted, never hidden.
- GARCH-HS naive fallbacks on walk-forward test rows: 26 / 1,396
  (1.86%), counted, never hidden.
- Event frame: 1,875 earnings events (t0 2022-01-03..2026-07-09);
  88 events past the price calendar end dropped, never labeled.
- Dev events: 1,874 (t0 <= 2026-06-30); 1,675 after dropping 199 rows
  with NaN features (insufficient trailing history for the 66-day
  leg); beta_252 fallbacks: 0.
- Test events (2026-07-01..2026-08-31): 1 labelable row; no
  test-period evaluation was run (GO-gated; see kill criterion).
- Confirmation rows (t0 >= 2026-09-01): never read, scored, or
  inspected; `periods.check_no_confirmation` enforced on every frame
  build, feature build, and evaluation call.

## Implementation notes (not deviations)

1. EARNINGS CACHE REUSE: the earnings download cache
   (`data/tailq_earnings.parquet`) is a byte copy of the EVENTVOL
   cache (`data/eventvol_earnings.parquet`); the download pipeline
   (`eventvol.earnings.download_earnings`, limit=32) is identical,
   so no re-download was needed. Caches are local resume aids and
   are never committed.
2. REPORTING PLUMB: after the first evaluation run, the runner was
   extended to persist pooled per-row forecasts
   (`data/tailq_pooled.parquet`, local, not committed) and to
   compute the pre-registered DSR honesty metric. The evaluation
   itself is fully deterministic (seed 7 everywhere, LightGBM
   deterministic=True); the re-run reproduced every fold number
   exactly, confirming the verdict is not an artifact of the
   re-run. The frozen design (arms, features, hyperparameters,
   metric, rule) is untouched.
3. CAViaR per-row fits are deterministic in their inputs
   (differential_evolution seed=7, fixed maxiter); the 20 fallbacks
   are short-window/degenerate fits per the frozen spec.

## Deviations

None.

## Kill criterion acknowledgment

Per the frozen kill criterion (docs/TAILQ_PREREGISTRATION.md): the
primary comparison is NO-GO on pinball (mean paired differential
+0.000250, 95% CI [-0.000882, +0.001382], lower bound not above
zero), so the TAILQ campaign is DEAD. It is not revived by trying
other models, other features, other quantiles, other horizons, other
event sets, or other metrics. No test-period evaluation was run (the
single authorized test eval is GO-gated). No confirmation data was
touched for any purpose. Any follow-up is a SEPARATELY
pre-registered campaign; nothing about TAILQ's data, features, or
tuning carries a gate claim into it.
