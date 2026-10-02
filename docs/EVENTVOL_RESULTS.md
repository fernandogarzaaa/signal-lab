# EVENTVOL campaign results: event-window volatility (working document)

Pre-registration: `docs/EVENTVOL_PREREGISTRATION.md` (frozen, binding). This document is the working record; the committed results PR carries the final version.

## Configuration

- Challenger: LightGBM (n_estimators=300, lr=0.05, leaves=31, min_child=100, feat_frac=0.8, bag_frac=0.8, l2=1.0, seed=7) on HAR(31) + VIX(4) + event(4) features; hyperparameters frozen, no selection
- Primary baseline: iv_proxy = beta(i,t0) x VIX(t0)/100/sqrt(252) (the market's forward-looking forecast; per-stock historical IV is not obtainable from yfinance, see pre-registration STEP 0)
- Secondary baselines: garch11 (scale-corrected 5-day term structure; fit failures fall back to naive persistence), har_vix (ridge alpha=1.0 on HAR+VIX features)
- Sanity arm: naive (trailing realized_vol_5d)
- Validation: 5-fold expanding purged walk-forward on t0, seed 7, min_train 50, purge on [t0, t0+5] overlap, 5-trading-day embargo
- Primary metric: QLIKE on variance (lower is better); second GO conjunct: challenger survives the 95% Model Confidence Set (Hansen-Lunde-Nason T_R, day-block stationary bootstrap, B=5000)
- Dev event rows: 1874 (1714 after NaN-feature drop)
- GARCH naive fallbacks on test rows: 51 / 1421
- iv_proxy fallbacks: {'n_beta_fallback': 30, 'n_vix_fallback': 0, 'n_rows': 1421}

## Per-fold QLIKE (variance)

| fold | test range | n_train | n_test | purged | embargoed | QLIKE challenger | QLIKE iv_proxy | QLIKE garch11 | QLIKE har_vix | QLIKE naive | d = iv - champ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-10-28..2023-07-26 | 259 | 279 | 34 | 0 | 0.647105 | 0.533723 | 0.696816 | 156.288047 | 2.045601 | -0.113382 |
| 2 | 2023-07-27..2024-04-24 | 544 | 287 | 28 | 0 | 0.820893 | 1.223952 | 1.447980 | 1.317386 | 9.242406 | +0.403059 |
| 3 | 2024-04-25..2025-01-27 | 811 | 284 | 25 | 23 | 2.946099 | 1.217073 | 1.920275 | 121.969861 | 5.330842 | -1.729026 |
| 4 | 2025-01-28..2025-10-16 | 1074 | 285 | 16 | 53 | 0.654630 | 0.694697 | 0.858578 | 411.728764 | 5.124104 | +0.040066 |
| 5 | 2025-10-17..2026-06-24 | 1336 | 286 | 12 | 80 | 0.798917 | 0.954909 | 0.977592 | 0.651725 | 2.561733 | +0.155992 |

## Verdict computation (frozen decision rule)

- d_k = QLIKE(iv_proxy) - QLIKE(challenger) per fold: -0.113382, +0.403059, -1.729026, +0.040066, +0.155992
- mean(d) = -0.248658, se = 0.379581, df = 4.0, t(0.975) = 2.7764
- 95% two-sided Student-t CI on mean(d): [-1.302543, +0.805226]
- Diebold-Mariano on fold-level QLIKE differentials (paired-t form): stat = -0.6551, two-sided p = 0.5482, n = 5.0
- 95% MCS survivors: ['challenger', 'iv_proxy']; eliminated (in order): [('har_vix', 0.02559488102379524), ('naive', 0.020195960807838434), ('garch11', 0.015996800639872025)]
- challenger in MCS: True
- GO iff mean(d) > 0 AND CI lower bound > 0 AND challenger in MCS: **NO-GO**

## Secondary metrics (per fold, no gate authority)

| fold | MSE-var challenger | MSE-var iv_proxy | MAE-vol challenger | MAE-vol iv_proxy |
| --- | --- | --- | --- | --- |
| 1 | 0.00000197 | 0.00000191 | 0.009922 | 0.008240 |
| 2 | 0.00000513 | 0.00000556 | 0.010293 | 0.010038 |
| 3 | 0.00000416 | 0.00000473 | 0.011156 | 0.011142 |
| 4 | 0.00000243 | 0.00000264 | 0.010393 | 0.008939 |
| 5 | 0.00000472 | 0.00000511 | 0.011754 | 0.011370 |

## Selection honesty (DSR, reported, not a gate)

per-row QLIKE gains over naive as pseudo-returns, freq=1; reported only, not a gate

- iv_proxy: Sharpe = +0.0841, DSR = 0.8567, likely_false_discovery = True
- garch11: Sharpe = +0.0793, DSR = 0.7974, likely_false_discovery = True
- challenger: Sharpe = +0.0765, DSR = 0.7232, likely_false_discovery = True
- har_vix: Sharpe = -0.0512, DSR = 0.0000, likely_false_discovery = True

## Deviations

- IV-PROXY VARIANCE DECOMPOSITION (deviation from the pre-registration's literal text): docs/EVENTVOL_PREREGISTRATION.md wrote iv_proxy = beta x VIX/100/sqrt(252) with the beta = 1.0 fallback specified only for <60 trailing observations. Two problems made the literal baseline degenerate by construction. (1) A non-positive trailing beta makes the forecast negative, and the 1e-6 positivity floor then turns it into a degenerate near-zero forecast that QLIKE punishes astronomically (per-row QLIKE up to ~8.7e8 in the first voided run). (2) For low-beta stocks (utilities, staples: beta ~ 0.0-0.05) the literal formula prices only the systematic leg and underforecasts total volatility by an order of magnitude (second voided run: iv_proxy fold QLIKE 140-597 vs challenger 0.65-2.95). Implemented instead: iv_proxy = sqrt((beta x VIX_d)^2 + idio_var), the textbook market-model variance decomposition, with (beta, idio_var) from the trailing 252-day regression vs SPY; uninformative regressions (<60 obs, beta <= 0, non-finite, degenerate window) fall back to beta = 1.0, idio_var = 0 (raw VIX), counted. Rationale: the pre-reg's stated intent is 'the market's forward-looking vol forecast or the closest honest proxy'; the decomposition is that proxy, and beta = 1.0 was already the established fallback for uninformative beta estimates. The correction strengthens the baseline (it can only make GO harder). Both earlier evaluation runs are VOID and discarded; only this run counts.
- EVENTS PAST THE PRICE CALENDAR END: the pre-registration freezes prices at 2022-01-01..2026-07-15 and drops rows with partial label windows. Earnings events dated past the last trading day cannot form any label window; they are dropped with a logged count (not a silent filter): build_event_frame drops them before t0 mapping. No test or confirmation rows are affected (the calendar ends 2026-07-15).
