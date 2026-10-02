# VOL campaign: NEGATIVE RESULT (NO-GO, campaign dead)

**Pre-registration:** `docs/VOL_PREREGISTRATION.md` (frozen, binding; merged
as PR #51, commit e5569c6). Target: `realized_vol_5d` = sample std
(ddof=1) of daily log returns over (t0, t0+5], decimal, daily.
Baseline: GARCH(1,1) by MLE on trailing 252 trading days, analytic 5-day
term structure. Arms: ridge (alpha=1.0, standardized) and LightGBM
(n_estimators=300, lr=0.05, leaves=31, min_child=100, feat_frac=0.8,
bag_frac=0.8, l2=1.0, seed=7) on 31 HAR-style point-in-time price
features + SPY legs. Metric: QLIKE on variance (lower is better).
Validation: 5-fold expanding purged walk-forward on t0, seed 7,
min_train 50, purge j==k, 5-trading-day embargo. Universe: S&P 100 (101)
+ SPY = 102 tickers, frozen in `data/universe_vol.csv`. Prices: yfinance
2022-01-01..2026-07-15.

## Verdict: NO-GO

The frozen decision rule: GO iff mean paired QLIKE differential
d = QLIKE(garch11) - QLIKE(champion) across folds satisfies mean(d) > 0
AND the lower bound of the 95% two-sided Student-t CI (df = n_folds - 1)
is > 0. Otherwise NO-GO.

### Per-fold QLIKE (variance)

| fold | test range | n_train | n_test | purged | embargoed | QLIKE champion (lightgbm) | QLIKE garch11 | QLIKE naive | d = garch - champ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-10-31..2023-07-27 | 18100 | 18500 | 500 | 0 | 0.546528 | 0.696149 | 1.599077 | +0.149621 |
| 2 | 2023-07-28..2024-04-23 | 36600 | 18600 | 500 | 0 | 0.613764 | 0.600372 | 2.041675 | -0.013392 |
| 3 | 2024-04-24..2025-01-15 | 54700 | 18480 | 500 | 500 | 0.685183 | 0.657060 | 2.172974 | -0.028122 |
| 4 | 2025-01-16..2025-10-08 | 72673 | 18625 | 505 | 1002 | 0.754244 | 0.785199 | 2.043785 | +0.030955 |
| 5 | 2025-10-09..2026-06-30 | 90788 | 18462 | 510 | 1507 | 0.600885 | 0.519340 | 1.759039 | -0.081545 |

### Verdict computation (step by step)

- d_k = QLIKE(garch11) - QLIKE(champion) per fold:
  +0.149621, -0.013392, -0.028122, +0.030955, -0.081545
- mean(d) = +0.011503
- se(d) = 0.038925 (sample std, ddof=1, divided by sqrt(5))
- df = 4; t(0.975, 4) = 2.7764
- 95% two-sided Student-t CI on mean(d):
  [+0.011503 - 2.7764 x 0.038925, +0.011503 + 2.7764 x 0.038925]
  = [-0.096569, +0.119576]
- mean(d) > 0 is TRUE (+0.011503), but the CI lower bound > 0 is FALSE
  (-0.096569). A narrow miss is NO-GO; the bar is not moved.
- Diebold-Mariano on fold-level QLIKE loss differentials (paired-t form):
  stat = +0.2955, two-sided p = 0.7823, n = 5. No significant difference.
- **Verdict: NO-GO.**

The champion beats the naive persistence baseline decisively on every
fold (QLIKE 0.55-0.75 vs 1.60-2.17) but does not beat the GARCH(1,1)
baseline: 2 folds won, 3 lost, mean differential +0.0115 with a CI
spanning zero. HAR-style price features + LightGBM do not demonstrably
improve on a correctly scaled GARCH(1,1) for 5-day realized volatility on
this universe and period.

## Selection (how the champion was chosen)

Single purged train/validation split inside dev (validation block
2025-08-15..2026-06-30; 88,419 train / 22,253 valid rows; 595 train rows
purged against the validation range). No walk-forward in selection.

- ridge validation QLIKE: 0.608419
- LightGBM validation QLIKE: 0.590065
- Champion: **lightgbm** (hyperparameters pre-declared in code before the
  run; see `src/signal_lab/vol/models.py`).

DSR honesty metric (reported, not a gate; per-row QLIKE gains over naive
as pseudo-returns, freq=1):

- lightgbm: Sharpe +0.1466, DSR 1.0000, likely_false_discovery = False
- ridge: Sharpe +0.1433, DSR 1.0000, likely_false_discovery = False

## Secondary metrics (per fold, no gate authority)

| fold | MSE-var champion | MSE-var garch11 | MAE-vol champion | MAE-vol garch11 |
| --- | --- | --- | --- | --- |
| 1 | 0.00000037 | 0.00000047 | 0.007913 | 0.007853 |
| 2 | 0.00000048 | 0.00000051 | 0.005822 | 0.006524 |
| 3 | 0.00000056 | 0.00000085 | 0.006702 | 0.007146 |
| 4 | 0.00000139 | 0.00000167 | 0.007625 | 0.009273 |
| 5 | 0.00000106 | 0.00000112 | 0.007491 | 0.007927 |

## Fit failures and effective row counts

- GARCH(1,1) naive fallbacks on walk-forward test rows: 5,166 / 92,667
  (5.57%). Fallbacks are short estimation windows (early dev, post-listing
  tickers GEV/SNDK) and non-convergent fits; all fell back to trailing
  realized_vol_5d persistence and are counted here, never hidden.
- Dev rows: 113,511 (t0 <= 2026-06-30); 111,267 after dropping 2,244 rows
  with NaN features (insufficient trailing history for the 22-day leg).
- Test rows (2026-07-01..2026-08-31): 510, untouched (no test evaluation
  was run; see kill criterion).
- Confirmation rows (t0 >= 2026-09-01): never read, scored, or inspected;
  `periods.check_no_confirmation` enforced on every frame build, split,
  and selection call.

## Deviations

1. BASELINE SCALE (deviation from the pre-registration's literal text):
   docs/VOL_PREREGISTRATION.md writes the GARCH(1,1) forecast as the 5-day
   term-structure variances "summed over h=1..5, square-rooted". Taken
   literally that is sqrt(sum of 5 daily variances) = 5-day cumulative
   volatility, sqrt(5) ~= 2.24x larger than the frozen target
   realized_vol_5d, which is defined (docs/TARGETS_AND_FEATURES.md v2.0,
   units "decimal, daily") as the sample std (ddof=1) of daily log returns
   over (t0, t0+5]. The literal reading puts the baseline on the wrong
   scale by construction (even a perfect-shape forecast scores QLIKE
   ~= 0.81 on scale alone; the literal-scale run showed GARCH QLIKE
   1.2-1.4 vs champion 0.55-0.75), which would make the
   champion-vs-baseline comparison meaningless. Implemented instead:
   sqrt(mean of the 5 h-step variances), i.e. the term-structure forecast
   of expected daily variance over the window, square-rooted to the
   target's daily scale - the same scale as the naive arm (trailing
   realized_vol_5d). Rationale: the precise target definition governs the
   forecast scale; the looser baseline phrasing is read as the
   term-structure aggregation step, not a scale change. The first
   evaluation run used the literal sqrt(sum); its GO verdict is VOID and
   discarded - only the scale-corrected run above counts.
2. TEXT ARMS SKIPPED (Phase 4, exploratory): the GDELT DOC API probe
   (query "("Apple Inc. stock" OR "AAPL stock") sourcelang:english" for
   2026-06-01, run 2026-10-02T11:05:20Z) returned 0 articles without
   error, but the first substantive quarterly query ("("AAPL stock")
   sourcelang:english", 2026-04-01..2026-06-30, attempted 2026-10-02
   ~11:07Z) got HTTP 429 Too Many Requests on all 4 attempts (client
   backoff 15s/120s/120s) and raised
   gdelt_client.errors.RateLimitError - the same hard throttle as gate-2.
   No proxies or rate-limit evasion attempted. The TF-IDF +
   Loughran-McDonald text arms are not run in this campaign.

## Kill criterion acknowledgment

Per the frozen kill criterion (docs/VOL_PREREGISTRATION.md): the primary
comparison is NO-GO, so the VOL campaign is DEAD. It is not revived by
trying other models, other features, other horizons, or other metrics.
No test-period evaluation was run (the single authorized test eval is GO-
gated). No confirmation data was touched for any purpose. Any follow-up
(e.g. the pre-registered next targets, event magnitude |excess_3d| or
drawdown risk) is a SEPARATELY pre-registered campaign; nothing about
VOL's data, features, or tuning carries a gate claim into it.
