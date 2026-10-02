# ATTRVOL campaign: NEGATIVE RESULT (NO-GO, campaign dead)

**Pre-registration:** `docs/ATTRVOL_PREREGISTRATION.md` (frozen, binding;
merged as PR #70, commit 486585d4). This is BET 2 (SPECULATIVE) from
`docs/SYNTHESIS_BETS.md`: attribution structure (driver concentration
from a market/idiosyncratic decomposition + price-path analogue-implied
volatility) as a predictor of post-event realized volatility. Target:
`realized_vol_5d` = sample std (ddof=1) of daily log returns over
(t0, t0+5], decimal, daily, on trailing top-decile |abn_ret| EVENT rows
only (|abn_ret(t0)| >= 90th percentile of the trailing 252 trading days
strictly before t0, min 60 observations; abn_ret = r_i - r_SPY).
Arms: har_vix_attr (challenger: ridge alpha=1.0, standardized, on 34
frozen HAR+VIX features + 5 frozen attribution features: driver_H,
idio_share, ret_x_H, analogue_vol_mean, analogue_vol_std) vs har_vix
(PRIMARY baseline: same ridge on the 34 HAR+VIX features alone) vs
garch11 (SECONDARY baseline: scale-corrected 5-day term structure, fit
failures fall back to naive persistence, counted) vs naive (sanity).
Metric: QLIKE on variance (lower is better). Validation: 5-fold
expanding purged walk-forward on t0 over event rows, seed 7, min_train
50, purge j==k, 5-trading-day embargo. Universe: S&P 100 (101) + SPY =
102 tickers, frozen in `data/universe_vol.csv`. Prices: yfinance
2022-01-01..2026-07-15. All features price-derived; no GDELT anywhere
in this campaign.

## Verdict: NO-GO

The frozen decision rule: GO iff mean paired QLIKE differential
d = QLIKE(har_vix) - QLIKE(har_vix_attr) across folds (over event-day
test rows) satisfies mean(d) > 0 AND the lower bound of the 95%
two-sided Student-t CI (df = n_folds - 1) is > 0. Otherwise NO-GO.

### Per-fold QLIKE (variance, event-day test rows)

| fold | test range | n_train | n_test | purged | embargoed | QLIKE har_vix_attr | QLIKE har_vix | QLIKE garch11 | QLIKE naive | d = harvix - challenger | d2 = garch - challenger |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2023-03-15..2024-02-07 | 1902 | 1942 | 53 | 0 | 0.701677 | 0.605741 | 0.553045 | 0.821799 | -0.095936 | -0.148632 |
| 2 | 2024-02-08..2024-09-11 | 3828 | 1955 | 69 | 0 | 49.543414 | 49.088867 | 0.697457 | 1.188429 | -0.454547 | -48.845957 |
| 3 | 2024-09-12..2025-04-09 | 5724 | 1978 | 77 | 51 | 0.835467 | 0.807458 | 0.875990 | 1.217647 | -0.028008 | +0.040523 |
| 4 | 2025-04-10..2026-01-22 | 7537 | 1916 | 205 | 88 | 0.707180 | 0.612610 | 0.575006 | 0.899718 | -0.094570 | -0.132174 |
| 5 | 2026-01-23..2026-06-30 | 9530 | 1942 | 58 | 158 | 0.692147 | 0.651433 | 0.490048 | 0.783441 | -0.040713 | -0.202099 |

### Verdict computation (step by step)

- d_k = QLIKE(har_vix) - QLIKE(har_vix_attr) per fold:
  -0.095936, -0.454547, -0.028008, -0.094570, -0.040713
- mean(d) = -0.142755
- se(d) = 0.079154 (sample std, ddof=1, divided by sqrt(5))
- df = 4; t(0.975, 4) = 2.7764
- 95% two-sided Student-t CI on mean(d):
  [-0.142755 - 2.7764 x 0.079154, -0.142755 + 2.7764 x 0.079154]
  = [-0.362522, +0.077012]
- mean(d) > 0 is FALSE (-0.142755). A narrow miss is NO-GO; the bar
  is not moved.
- Diebold-Mariano on fold-level QLIKE loss differentials (paired-t
  form): stat = -1.8035, two-sided p = 0.1456, n = 5. No significant
  difference.
- **Verdict: NO-GO.**

### The verdict does not hinge on fold 2

Fold 2's blowup (both ridge arms at QLIKE ~49) is a genuine model
pathology on event days: on a subset of event rows the frozen ridge
arms extrapolate to floored 1e-4 forecasts, and QLIKE explodes. The
attribution challenger blows up marginally harder (49.54 vs 49.09),
but the pathology hits the HAR+VIX baseline too: it is not caused by
the attribution features. Excluding fold 2, the remaining folds give
mean(d) = -0.064807, se = 0.017771, 95% t CI (df=3) =
[-0.121361, -0.008253]: the CI is ENTIRELY negative. The attribution
features do not merely fail to add; on the non-blowup folds they make
the forecast significantly worse than HAR+VIX alone.

## Secondary comparisons (reported, no gate authority)

- Challenger vs garch11: d2_k = -0.148632, -48.845957, +0.040523,
  -0.132174, -0.202099; mean(d2) = -9.857668,
  95% CI [-36.920115, +17.204779]; DM stat = -1.0113, p = 0.3691.
- 95% Model Confidence Set (Hansen-Lunde-Nason T_R, day-block
  stationary bootstrap B=5000, 9,733 pooled dev event rows):
  eliminated har_vix_attr (p=0.0002), har_vix (p=0.0002), naive
  (p=0.0002); survivor = {garch11}. The challenger is the FIRST arm
  eliminated.

## Secondary metrics (per fold, no gate authority)

| fold | MSE-var challenger | MSE-var har_vix | MSE-var garch11 | MAE-vol challenger | MAE-vol har_vix | MAE-vol garch11 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.00000029 | 0.00000037 | 0.00000036 | 0.005919 | 0.006041 | 0.007356 |
| 2 | 0.00000119 | 0.00000118 | 0.00000375 | 0.007324 | 0.007250 | 0.007803 |
| 3 | 0.00000203 | 0.00000204 | 0.00000280 | 0.008865 | 0.009180 | 0.010008 |
| 4 | 0.00000194 | 0.00000184 | 0.00000199 | 0.008169 | 0.008051 | 0.009898 |
| 5 | 0.00000142 | 0.00000145 | 0.00000195 | 0.008529 | 0.008643 | 0.009203 |

## Selection honesty (DSR, reported, not a gate)

Per-row QLIKE gains of the two tried ridge arms over naive as
pseudo-returns, freq=1, pooled dev event test rows:

- har_vix: Sharpe -0.0135, DSR 0.0009, likely_false_discovery = True
- har_vix_attr: Sharpe -0.0137, DSR 0.0006, likely_false_discovery = True

## Fit failures and effective row counts

- GARCH(1,1) naive fallbacks on walk-forward event test rows: 114 /
  9,733 (1.17%). All fallbacks fell back to trailing realized_vol_5d
  persistence and are counted here, never hidden.
- Dev event rows: 11,688 (t0 <= 2026-06-30); 11,688 after the
  NaN-feature drop (0 dropped: every dev event row featurized
  cleanly, including >= 3 valid analogues).
- Beta fallbacks (trailing_beta < 60 obs): 0 / 11,806 event rows.
- Test rows (2026-07-01..2026-08-31): untouched (no test evaluation
  was run; see kill criterion).
- Confirmation rows (t0 >= 2026-09-01): never read, scored, or
  inspected; `periods.check_no_confirmation` enforced on every frame
  build, split, and evaluation call.

## Deviations

None. The frame rebuilt from the merged build code was verified
byte-identical (md5 408dfabb30ce33a86eafdb8640964afb) to the frame
built during development, proving the evaluated code is the committed
code. One post-merge build fix (PR #72) silenced an expected
RuntimeWarning on all-NaN path windows for pre-listing tickers; it is
numerically identical (warning suppression only) and is not a spec
change. The pre-registration's "trailing 252 trading days ending at
t0" for the event quantile was implemented as the 252 days strictly
before t0 (shifted rolling quantile; the event day never enters its
own threshold); both readings are point-in-time and the strict
reading is the standard causal one.

## Diagnosis (why the bet failed)

The data supports the bet memo's own stated failure modes, and then
some. On non-blowup folds the challenger underperforms HAR+VIX on
every fold (d_k < 0 for all k), significantly so excluding fold 2:
the two-leg CAPM decomposition appears too coarse to capture true
driver structure, and the analogue features add noise rather than
signal, consistent with "analogues by price path match noise, not
mechanism." The event-day return magnitude itself (already in HAR via
rv_1d) appears to saturate what is predictable about post-event vol.
Additionally, restricting evaluation to event days exposes a ridge
extrapolation pathology that the full-sample campaigns masked: on
fold 2 both ridge arms collapse to floored forecasts while GARCH
stays sane (0.70), which is why the 95% MCS eliminates both ridge
arms and keeps only the GARCH null.

## Kill criterion acknowledgment

Per the frozen kill criterion (docs/ATTRVOL_PREREGISTRATION.md): the
primary comparison is NO-GO, so the ATTRVOL campaign is DEAD. It is
not revived by trying other models, other features, other horizons,
or other metrics. No test-period evaluation was run (the single
authorized test eval is GO-gated). No confirmation data was touched
for any purpose. The pre-registered next bet is CONFMAG
(docs/SYNTHESIS_BETS.md), SEPARATELY pre-registered; nothing about
ATTRVOL's data, features, or tuning carries a gate claim into it.
