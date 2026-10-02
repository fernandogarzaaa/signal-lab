# DISPVOL campaign: NEGATIVE RESULT (NO-GO, campaign dead)

**Pre-registration:** `docs/DISPVOL_PREREGISTRATION.md` (frozen, binding;
merged as PR #67, commit 1d8bbae). This is BET 1 (SPECULATIVE) from
`docs/SYNTHESIS_BETS.md`: cross-sectional dispersion predicts
individual realized volatility. Target: `realized_vol_5d` = sample std
(ddof=1) of daily log returns over (t0, t0+5], decimal, daily, on ALL
ticker-days. Challenger: har_vix_disp = ridge (alpha = 1.0,
standardized) on the 34 frozen HAR+VIX features plus 6 frozen
cross-sectional dispersion features (disp_level, disp_5d_change,
disp_z60, each also interacted with the stock's own rv_1d; 40 features
total). Primary baseline: har_vix = ridge (alpha = 1.0, standardized)
on the 34 HAR+VIX features alone (the HARVIX challenger definition,
unchanged). Secondary baseline: garch11 (the same scale-corrected
GARCH(1,1) implementation as VOL/HARVIX: MLE on trailing 252 trading
days, analytic 5-day term structure averaged to the target's daily
scale; fit failures fall back to naive persistence, counted). Sanity
arm: naive (trailing realized_vol_5d). Metric: QLIKE on variance
(lower is better), 95% CI via the paired-t form (documented
HAC-equivalent on non-overlapping fold blocks), Diebold-Mariano, 95%
Model Confidence Set (reported, not a gate). Validation: 5-fold
expanding purged walk-forward on t0, seed 7, min_train 50, purge j==k,
5-trading-day embargo. Universe: S&P 100 (101) + SPY = 102 tickers,
frozen in `data/universe_vol.csv`. Prices: yfinance 2022-01-01..2026-07-15
(reused VOL download; quality gates passed). VIX: ^VIX + ^VIX3M daily
closes reused from the HARVIX cache (1,136 rows each). No GDELT
anywhere in this campaign.

## Verdict: NO-GO

The frozen decision rule: GO iff mean paired QLIKE differential
d = QLIKE(har_vix) - QLIKE(har_vix_disp) across folds satisfies
mean(d) > 0 AND the lower bound of the 95% two-sided Student-t CI
(df = n_folds - 1) is > 0. Otherwise NO-GO.

### Per-fold QLIKE (variance)

| fold | test range | n_train | n_test | purged | embargoed | QLIKE har_vix_disp | QLIKE har_vix | QLIKE garch11 | QLIKE naive | d = harvix - challenger | d2 = garch - challenger |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-12-20..2023-09-06 | 17400 | 17800 | 500 | 0 | 0.572268 | 0.570694 | 0.563762 | 1.698339 | -0.001574 | -0.008506 |
| 2 | 2023-09-07..2024-05-22 | 35200 | 17918 | 500 | 0 | 0.902833 | 0.901771 | 0.624821 | 2.167574 | -0.001062 | -0.278012 |
| 3 | 2024-05-23..2025-02-05 | 52613 | 17776 | 505 | 500 | 0.759259 | 0.761347 | 0.696948 | 2.267553 | +0.002087 | -0.062311 |
| 4 | 2025-02-06..2025-10-17 | 69884 | 17925 | 505 | 1005 | 15.845790 | 13.089495 | 0.735470 | 1.841576 | -2.756295 | -15.110320 |
| 5 | 2025-10-20..2026-06-30 | 87299 | 17748 | 510 | 1510 | 0.579349 | 0.607108 | 0.516799 | 1.771612 | +0.027759 | -0.062550 |

Dev rows: 113,511 (t0 2022-01-03..2026-06-30); 107,067 after the
NaN-feature drop (6,444 dropped, of which 6,402 are the dispersion
65-day burn-in). GARCH naive fallbacks on test rows: 1,666 / 89,167
(1.87%).

### Verdict computation (step by step)

- d_k = QLIKE(har_vix) - QLIKE(har_vix_disp) per fold:
  -0.001574, -0.001062, +0.002087, -2.756295, +0.027759
- mean(d) = -0.545817
- se(d) = 0.552646 (sample std, ddof=1, divided by sqrt(5))
- df = 4; t(0.975, 4) = 2.7764
- 95% two-sided Student-t CI on mean(d):
  [-0.545817 - 2.7764 x 0.552646, -0.545817 + 2.7764 x 0.552646]
  = [-2.080209, +0.988575]
- mean(d) > 0 is FALSE (-0.545817). The CI straddles zero.
- Diebold-Mariano on fold-level QLIKE loss differentials (paired-t
  form, HAC-equivalent on non-overlapping blocks): stat = -0.9876,
  two-sided p = 0.3792, n = 5. No significant difference.
- GO iff mean(d) > 0 AND CI lower bound > 0: **NO-GO**.

### Secondary comparison: challenger vs garch11 (reported, no gate authority)

- d2_k = QLIKE(garch11) - QLIKE(har_vix_disp) per fold:
  -0.008506, -0.278012, -0.062311, -15.110320, -0.062550
- mean(d2) = -3.104340, se = 3.001852
- 95% CI (df=4): [-11.438817, +5.230137]
- DM: stat = -1.0341, two-sided p = 0.3595, n = 5.
- The challenger does not beat the GARCH(1,1) null either; GARCH is
  the 95% Model Confidence Set's sole survivor (below).

### Model Confidence Set (95%, reported, not a gate)

- Survivors: garch11 (n_rows=89167, B=5000, alpha=0.05, seed=7)
- Eliminated (in order): har_vix_disp (p=0.0002), har_vix (p=0.0002),
  naive (p=0.0002)
- The challenger is the FIRST arm eliminated. Both ridge arms are
  statistically indistinguishable from each other and both are
  rejected against the GARCH null.

### Secondary metrics (per fold, no gate authority)

| fold | MSE-var challenger | MSE-var har_vix | MSE-var garch11 | MAE-vol challenger | MAE-vol har_vix | MAE-vol garch11 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.00000039 | 0.00000039 | 0.00000036 | 0.006874 | 0.006838 | 0.007288 |
| 2 | 0.00000055 | 0.00000055 | 0.00000054 | 0.005579 | 0.005582 | 0.006525 |
| 3 | 0.00000079 | 0.00000079 | 0.00000112 | 0.007008 | 0.007011 | 0.007416 |
| 4 | 0.00000152 | 0.00000152 | 0.00000148 | 0.007778 | 0.007791 | 0.009184 |
| 5 | 0.00000108 | 0.00000108 | 0.00000113 | 0.007823 | 0.007702 | 0.007898 |

### Selection honesty (DSR, reported, not a gate)

Per-row QLIKE gains of the tried arms over naive as pseudo-returns,
freq=1, pooled dev test rows; reported only, not a gate.

- har_vix: Sharpe = -0.0056, DSR = 0.0053, likely_false_discovery = True
- har_vix_disp: Sharpe = -0.0071, DSR = 0.0002, likely_false_discovery = True

### What the numbers say

On folds 1, 2, 3, and 5 the challenger and the HAR+VIX baseline are
nearly identical (|d| <= 0.028): the six dispersion features add
nothing the 34 HAR+VIX features do not already capture. Fold 4 is the
same pathology that killed HARVIX (the 2025-04-09 VIX +12.11 spike
regime): both ridge arms blow up, and the dispersion challenger blows
up HARDER (QLIKE 15.85 vs 13.09), consistent with the bet memo's
stated failure mode that dispersion is a coincident rather than
leading indicator: it spikes during volatile regimes, which the HAR
legs already price, and the extra features give the ridge more rope
to extrapolate on the spike. Excluding fold 4, mean(d) = +0.006803
with a 95% CI of [-0.015576, +0.029183]: still straddling zero, still
NO-GO. There is no version of this comparison that clears the bar.

### Deviations

None. Target, features (including the exact point-in-time
dispersion construction and the 65-day burn-in), arms, horizon,
metric, universe, periods, and decision rule are exactly as
pre-registered. No test-period or confirmation data was touched at
any point; `periods.check_no_confirmation` was enforced on every
frame build, split, and evaluation call.

### Kill-criterion acknowledgment

The primary comparison is NO-GO, so per the frozen kill criterion
the DISPVOL campaign is DEAD. It is not revived by other models,
other features, other horizons, or other metrics. No test-period
evaluation was run (GO-gated). No confirmation data was touched. The
next bets (ATTRVOL, CONFMAG) require their own separate
pre-registrations; nothing about DISPVOL's data, features, or tuning
carries a gate claim into them.

### Artifacts

- `docs/dispvol_results.json`: full per-fold table, CIs, DM, MCS,
  DSR, verdict (committed).
- `src/signal_lab/dispvol/`: frozen feature + runner implementation
  (build PR).
- `tests/test_dispvol_features.py`, `tests/test_dispvol_pipeline.py`:
  15 synthetic tests (dispersion math, SPY exclusion, 50-ticker
  minimum, point-in-time truncation, no-future-info, burn-in,
  interaction math, fail-loud guards, end-to-end walk-forward smoke,
  confirmation enforcement).
