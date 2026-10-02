# HARVIX campaign: NEGATIVE RESULT (NO-GO, campaign dead)

**Pre-registration:** `docs/HARVIX_PREREGISTRATION.md` (frozen, binding;
merged as PR #61, commit 8d5b848). This is Candidate B from
docs/RESEARCH_SPRINT.md section 4: the literature's highest-conviction
volatility arm (Blair, Poon, and Taylor 2001; Pong, Shackleton, Taylor,
and Xu 2004), tested as a one-arm rematch on the VOL campaign's frozen
target. Target: `realized_vol_5d` = sample std (ddof=1) of daily log
returns over (t0, t0+5], decimal, daily, on ALL ticker-days (not
event-restricted). Challenger: har_vix = ridge (alpha = 1.0,
standardized) on the 31 VOL HAR-style point-in-time price features + 3
frozen VIX features (vix_level = VIX/100/sqrt(252); vix_5d_change =
VIX(t0) - VIX(t0 - 5 trading days); vix_slope = VIX3M/VIX - 1).
Primary baseline: garch11 (the same scale-corrected implementation as
VOL: MLE on trailing 252 trading days, analytic 5-day term structure
averaged to daily scale; fit failures fall back to naive persistence,
counted). Secondary baseline: har = ridge (alpha = 1.0, standardized)
on the 31 HAR features alone. Sanity arm: naive (trailing
realized_vol_5d). Metric: QLIKE on variance (lower is better), 95% CI
via the paired-t form (documented HAC-equivalent on non-overlapping
fold blocks), Diebold-Mariano, 95% Model Confidence Set (reported, not
a gate). Validation: 5-fold expanding purged walk-forward on t0, seed 7,
min_train 50, purge j==k, 5-trading-day embargo. Universe: S&P 100
(101) + SPY = 102 tickers, frozen in `data/universe_vol.csv`. Prices:
yfinance 2022-01-01..2026-07-15 (reused VOL download; quality gates
passed). VIX: ^VIX + ^VIX3M daily closes 2022-01-03..2026-07-14,
downloaded from yfinance at build time (fail-loud; both series verified
available, 1,136 rows each).

## Verdict: NO-GO

The frozen decision rule: GO iff mean paired QLIKE differential
d = QLIKE(garch11) - QLIKE(har_vix) across folds satisfies mean(d) > 0
AND the lower bound of the 95% two-sided Student-t CI (df = n_folds - 1)
is > 0. Otherwise NO-GO.

### Per-fold QLIKE (variance)

| fold | test range | n_train | n_test | purged | embargoed | QLIKE har_vix | QLIKE garch11 | QLIKE har | QLIKE naive | d = garch - harvix | d2 = har - harvix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-10-31..2023-07-27 | 18100 | 18500 | 500 | 0 | 0.570555 | 0.696149 | 0.545860 | 1.599077 | +0.125595 | -0.024694 |
| 2 | 2023-07-28..2024-04-23 | 36600 | 18600 | 500 | 0 | 0.740193 | 0.600372 | 0.626063 | 2.041675 | -0.139821 | -0.114130 |
| 3 | 2024-04-24..2025-01-15 | 54700 | 18480 | 500 | 500 | 0.711194 | 0.657060 | 0.740367 | 2.172974 | -0.054134 | +0.029173 |
| 4 | 2025-01-16..2025-10-08 | 72673 | 18625 | 505 | 1002 | 4.882932 | 0.785199 | 0.794081 | 2.043785 | -4.097733 | -4.088851 |
| 5 | 2025-10-09..2026-06-30 | 90788 | 18462 | 510 | 1507 | 0.614739 | 0.519340 | 0.609344 | 1.759039 | -0.095399 | -0.005395 |

### Verdict computation (step by step)

- d_k = QLIKE(garch11) - QLIKE(har_vix) per fold:
  +0.125595, -0.139821, -0.054134, -4.097733, -0.095399
- mean(d) = -0.852298
- se(d) = 0.812610 (sample std, ddof=1, divided by sqrt(5))
- df = 4; t(0.975, 4) = 2.7764
- 95% two-sided Student-t CI on mean(d):
  [-0.852298 - 2.7764 x 0.812610, -0.852298 + 2.7764 x 0.812610]
  = [-3.108466, +1.403869]
- mean(d) > 0 is FALSE (-0.852298). The CI straddles zero.
- Diebold-Mariano on fold-level QLIKE loss differentials (paired-t form,
  HAC-equivalent on non-overlapping blocks): stat = -1.0488, two-sided
  p = 0.3534, n = 5. No significant difference.
- **Verdict: NO-GO.**

The challenger loses to GARCH(1,1) on 4 of 5 folds. The fold-4
catastrophe (d = -4.10) dominates the mean, but the verdict does not
hinge on it: dropping fold 4, mean(d) over the remaining folds is
(0.125595 - 0.139821 - 0.054134 - 0.095399) / 4 = -0.040940, still
negative. HAR+VIX does not demonstrably improve on GARCH(1,1) for
per-stock 5-day realized volatility on this universe and period.

### Secondary comparison: har_vix vs har (reported, no gate authority)

- d2_k = QLIKE(har) - QLIKE(har_vix) per fold:
  -0.024694, -0.114130, +0.029173, -4.088851, -0.005395
- mean(d2) = -0.840780, 95% CI [-3.096260, +1.414701]
- DM: stat = -1.0350, p = 0.3591
- VIX does not add to the history-based model here: the challenger
  loses to plain HAR on 4 of 5 folds (the same fold-4 blowup drives the
  mean, and the ex-fold-4 mean is likewise negative). The literature's
  index-level finding (Blair-Poon-Taylor 2001) does not transfer to
  per-stock 5-day vol on this frame with this estimator.

### Model Confidence Set (95%, reported, not a gate)

Hansen-Lunde-Nason T_R, day-block stationary bootstrap, B=5000, mean
block 10 trading days, seed 7, over {har_vix, garch11, har, naive} on
92,667 pooled per-row dev QLIKE: eliminated naive (p=0.0002); final
round p=0.3603, no further elimination. Survivors: har_vix, garch11,
har. The MCS cannot separate the three serious arms; the primary rule
decides, and it says NO-GO.

## The fold-4 blowup (diagnosis, dev data)

Fold 4's har_vix QLIKE of 4.88 (vs 0.79 for GARCH) comes from a
handful of rows, not a broad failure. On 2025-04-09 the VIX printed a
+12.11-point 5-day spike (a market stress episode). The ridge arm,
whose fold-4 training data never saw that combination of a large VIX
spike with defensive-stock HAR features, extrapolated to deeply
negative raw forecasts that hit the 1e-4 floor. The 8 worst rows
(MDLZ, PG, KO, SO, DUK, PEP on 2025-04-09; UNH on 2025-04-16) carry
per-row QLIKE of 816 to 25,650 and contribute ~4.15 of the 4.88 fold
mean. This is the same genuine model pathology the EVENTVOL campaign
documented for ridge on HAR+VIX features (docs/EVENTVOL_NEGATIVE_RESULT.md,
"Notes on the secondary baselines"): a linear model with no
extrapolation guard is fragile to IV-regime observations outside its
training range. It is a model failure, not a code bug: the ridge fit
was correct per fold with frozen hyperparameters, the point-in-time
contract held, and the 1e-4 floor behaved as specified. The
pre-registered estimator is what was tested; its fragility is the
finding.

## Selection honesty

No model or hyperparameter selection ran in this campaign: all
estimators and feature lists were frozen in the pre-registration before
any data was touched. DSR honesty metric (reported, not a gate;
per-row QLIKE gains over naive as pseudo-returns, freq=1, pooled dev
test rows):

- har: Sharpe +0.0983, DSR 1.0000, likely_false_discovery = False
- har_vix: Sharpe +0.0036, DSR 0.0000, likely_false_discovery = True

## Secondary metrics (per fold, no gate authority)

| fold | MSE-var har_vix | MSE-var garch11 | MSE-var har | MAE-vol har_vix | MAE-vol garch11 | MAE-vol har |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.00000038 | 0.00000047 | 0.00000038 | 0.006193 | 0.007853 | 0.007380 |
| 2 | 0.00000051 | 0.00000051 | 0.00000049 | 0.005543 | 0.006524 | 0.006143 |
| 3 | 0.00000058 | 0.00000085 | 0.00000058 | 0.006759 | 0.007146 | 0.006659 |
| 4 | 0.00000165 | 0.00000167 | 0.00000259 | 0.007846 | 0.009273 | 0.008647 |
| 5 | 0.00000105 | 0.00000112 | 0.00000106 | 0.007682 | 0.007927 | 0.007574 |

## Fit failures and effective row counts

- GARCH(1,1) naive fallbacks on walk-forward test rows: 5,166 / 92,667
  (5.57%), counted, never hidden (same fallbacks as VOL; see
  implementation note below).
- Dev rows: 113,511 (t0 <= 2026-06-30); 111,267 after dropping 2,244
  rows with NaN features (insufficient trailing history for the 22-day
  legs; the VIX 22-day burn-in aligns exactly with the price burn-in,
  so the VIX features drop zero rows beyond VOL's drop).
- Test rows (2026-07-01..2026-08-31): untouched (no test evaluation was
  run; see kill criterion).
- Confirmation rows (t0 >= 2026-09-01): never read, scored, or
  inspected; `periods.check_no_confirmation` enforced on every frame
  build, split, and evaluation call.

## Implementation notes (not deviations)

1. GARCH PANEL MEMOIZATION: the garch11 baseline is the same
   implementation as VOL (`src/signal_lab/vol/garch.py`, unchanged),
   deterministic in its inputs. HARVIX's dev test-row set turned out
   to be exactly VOL's (92,667 identical (ticker, t0) rows: same
   frame, same NaN drop, same deterministic splits), so all 92,667
   forecasts were reused from VOL's cached panel
   (`data/vol_garch_panel.parquet`) via a one-off join script
   (`/tmp/harvix_panel_join.py`, not committed); zero rows needed new
   fits. Verified literally: every joined value is identical to VOL's
   panel. The pre-registration's baseline definition (implementation,
   252-day window, 5-day horizon, scale correction, fallback rule) is
   unchanged; this is memoization, not a baseline change.
2. ^VIX3M AVAILABILITY: verified at build time via a real yfinance
   download (1,136 rows, 2022-01-03..2026-07-14), as the
   pre-registration required; the term-structure slope uses live
   ^VIX3M closes.

## Deviations

None.

## Kill criterion acknowledgment

Per the frozen kill criterion (docs/HARVIX_PREREGISTRATION.md): the
primary comparison is NO-GO, so the HARVIX campaign is DEAD. It is not
revived by trying other models (e.g. an extrapolation-guarded
estimator), other features, other horizons, or other metrics. No
test-period evaluation was run (the single authorized test eval is
GO-gated). No confirmation data was touched for any purpose. Any
follow-up is a SEPARATELY pre-registered campaign; nothing about
HARVIX's data, features, or tuning carries a gate claim into it.
