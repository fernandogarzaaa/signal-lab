# EVENTVOL campaign: NEGATIVE RESULT (NO-GO, campaign dead)

**Pre-registration:** `docs/EVENTVOL_PREREGISTRATION.md` (frozen, binding;
merged as PR #56, commit e11daf2). Target: `realized_vol_5d` = sample std
(ddof=1) of daily log returns over (t0, t0+5], decimal, daily, on
earnings-event rows (one row per ticker per reported earnings date).
Primary baseline: iv_proxy = sqrt((beta x VIX_d)^2 + idio_var), the
market-model variance decomposition of index implied volatility
(the market's forward-looking forecast; per-stock historical IV is not
obtainable from yfinance, so the index-IV translation is the closest
honest proxy, reformulated as a deliberate design choice in the
pre-registration). Secondary baselines: garch11 (scale-corrected 5-day
term structure), har_vix (ridge alpha=1.0 on HAR+VIX features). Sanity
arm: naive (trailing realized_vol_5d). Challenger: LightGBM
(n_estimators=300, lr=0.05, leaves=31, min_child=100, feat_frac=0.8,
bag_frac=0.8, l2=1.0, seed=7; hyperparameters frozen, no selection) on
HAR(31) + VIX(4) + event(4) features (Ederington-Lee IV
creation/resolution cycle proxies: event-day return and absolute
return, trailing-quarter EPS surprise, trailing beta, VIX level /
5-day change / 22-day run-up / 3M-1M term-structure slope). Metric:
QLIKE on variance (lower is better). Validation: 5-fold expanding
purged walk-forward on t0, seed 7, min_train 50, purge on [t0, t0+5]
overlap, 5-trading-day embargo. Universe: S&P 100 (100 with earnings;
BRK.B and SPY have no earnings calendar) frozen in
`data/universe_vol.csv`. Prices: yfinance 2022-01-01..2026-07-15. VIX:
^VIX/^VIX3M same window. Earnings: yfinance earnings calendar, actuals
only (non-NaN Reported EPS).

## Verdict: NO-GO

The frozen decision rule: GO iff mean paired QLIKE differential
d = QLIKE(iv_proxy) - QLIKE(challenger) across folds satisfies
mean(d) > 0 AND the lower bound of the 95% two-sided Student-t CI
(df = n_folds - 1) is > 0 AND the challenger survives the 95% Model
Confidence Set. Otherwise NO-GO.

### Per-fold QLIKE (variance)

| fold | test range | n_train | n_test | purged | embargoed | QLIKE challenger | QLIKE iv_proxy | QLIKE garch11 | QLIKE har_vix | QLIKE naive | d = iv - champ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-10-28..2023-07-26 | 259 | 279 | 34 | 0 | 0.647105 | 0.533723 | 0.696816 | 156.288047 | 2.045601 | -0.113382 |
| 2 | 2023-07-27..2024-04-24 | 544 | 287 | 28 | 0 | 0.820893 | 1.223952 | 1.447980 | 1.317386 | 9.242406 | +0.403059 |
| 3 | 2024-04-25..2025-01-27 | 811 | 284 | 25 | 23 | 2.946099 | 1.217073 | 1.920275 | 121.969861 | 5.330842 | -1.729026 |
| 4 | 2025-01-28..2025-10-16 | 1074 | 285 | 16 | 53 | 0.654630 | 0.694697 | 0.858578 | 411.728764 | 5.124104 | +0.040066 |
| 5 | 2025-10-17..2026-06-24 | 1336 | 286 | 12 | 80 | 0.798917 | 0.954909 | 0.977592 | 0.651725 | 2.561733 | +0.155992 |

### Verdict computation (step by step)

- d_k = QLIKE(iv_proxy) - QLIKE(challenger) per fold:
  -0.113382, +0.403059, -1.729026, +0.040066, +0.155992
- mean(d) = -0.248658
- se(d) = 0.379581 (sample std, ddof=1, divided by sqrt(5))
- df = 4; t(0.975, 4) = 2.7764
- 95% two-sided Student-t CI on mean(d):
  [-0.248658 - 2.7764 x 0.379581, -0.248658 + 2.7764 x 0.379581]
  = [-1.302543, +0.805226]
- mean(d) > 0 is FALSE (-0.248658). The CI straddles zero.
- Diebold-Mariano on fold-level QLIKE loss differentials (paired-t form):
  stat = -0.6551, two-sided p = 0.5482, n = 5. No significant difference.
- 95% Model Confidence Set (Hansen-Lunde-Nason T_R, day-block stationary
  bootstrap, B=5000, mean block 10 trading days, seed 7) over
  {challenger, iv_proxy, garch11, har_vix, naive} on pooled per-row QLIKE:
  eliminated har_vix (p=0.0256), then naive (p=0.0202), then garch11
  (p=0.0160); final round p=0.5967, no further elimination.
  Survivors: challenger, iv_proxy. The challenger survives the MCS, so
  the second GO conjunct passes, but the primary conjunct fails.
- **Verdict: NO-GO.**

The challenger does not add to the market's forward-looking forecast:
it loses to iv_proxy by 0.25 QLIKE points on average across folds
(2 folds won, 3 lost), with a confidence interval spanning zero and a
DM p-value of 0.55. The result is consistent with the literature the
campaign was built on: implied volatility subsumes history-based
models (Blair, Poon, and Taylor 2001), and here iv_proxy also beats
garch11 on all 5 folds (0.53 vs 0.70, 1.22 vs 1.45, 1.22 vs 1.92,
0.69 vs 0.86, 0.95 vs 0.98). The market's price was the right bar, and
nothing in the challenger's event features cleared it.

## Selection honesty

No model or feature selection ran in this campaign: both ML arms use
frozen hyperparameters and frozen feature lists declared in the
pre-registration (the only selection in the program's history, ridge vs
LightGBM, was VOL's). DSR honesty metric (reported, not a gate;
per-row QLIKE gains over naive as pseudo-returns, freq=1):

- iv_proxy: Sharpe +0.0841, DSR 0.8567, likely_false_discovery = True
- garch11: Sharpe +0.0793, DSR 0.7974, likely_false_discovery = True
- challenger: Sharpe +0.0765, DSR 0.7232, likely_false_discovery = True
- har_vix: Sharpe -0.0512, DSR 0.0000, likely_false_discovery = True

## Secondary metrics (per fold, no gate authority)

| fold | MSE-var challenger | MSE-var iv_proxy | MAE-vol challenger | MAE-vol iv_proxy |
| --- | --- | --- | --- | --- |
| 1 | 0.00000038 | 0.00000033 | 0.010106 | 0.009919 |
| 2 | 0.00000166 | 0.00000306 | 0.013203 | 0.014904 |
| 3 | 0.00001247 | 0.00000202 | 0.022278 | 0.015052 |
| 4 | 0.00000068 | 0.00000066 | 0.009937 | 0.010160 |
| 5 | 0.00000101 | 0.00000141 | 0.011283 | 0.011921 |

## Fit failures and effective row counts

- Event frame: 1,874 earnings-event rows (100 tickers; BRK.B and SPY
  have no earnings calendar), t0 2022-01-03..2026-06-24. All 1,874 rows
  fall in dev (t0 <= 2026-06-30); 160 rows dropped for NaN features
  (insufficient trailing history for the 252-day beta / 22-day HAR legs
  on early-2022 events), leaving 1,714 rows for the walk-forward.
- GARCH(1,1) naive fallbacks on walk-forward test rows: 51 / 1,421
  (3.59%), counted, never hidden.
- iv_proxy fallbacks: 30 beta fallbacks (uninformative trailing
  regression: <60 obs, non-positive or non-finite beta, degenerate
  window) / 1,421; 0 VIX fallbacks.
- Test rows (2026-07-01..2026-08-31): 0 labelable (prices end
  2026-07-15, so no earnings event in the test window has a full
  5-day label window). No test-period evaluation was run; on NO-GO the
  kill criterion forbids it in any case.
- Confirmation rows (t0 >= 2026-09-01): never read, scored, or
  inspected; `periods.check_no_confirmation` enforced on every frame
  build, split, and evaluation call.

## Deviations

1. IV-PROXY VARIANCE DECOMPOSITION (deviation from the
   pre-registration's literal text): docs/EVENTVOL_PREREGISTRATION.md
   wrote iv_proxy = beta x VIX/100/sqrt(252) with the beta = 1.0 fallback
   specified only for <60 trailing observations. Two problems made the
   literal baseline degenerate by construction. (1) A non-positive
   trailing beta makes the forecast negative, and the 1e-6 positivity
   floor then turns it into a degenerate near-zero forecast that QLIKE
   punishes astronomically (per-row QLIKE up to ~8.7e8 in the first
   voided run). (2) For low-beta stocks (utilities, staples:
   beta ~ 0.0-0.05) the literal formula prices only the systematic leg
   and underforecasts total volatility by an order of magnitude (second
   voided run: iv_proxy fold QLIKE 140-597 vs challenger 0.65-2.95).
   Implemented instead: iv_proxy = sqrt((beta x VIX_d)^2 + idio_var),
   the textbook market-model variance decomposition, with (beta,
   idio_var) from the trailing 252-day regression vs SPY;
   uninformative regressions (<60 obs, beta <= 0, non-finite,
   degenerate window) fall back to beta = 1.0, idio_var = 0 (raw VIX),
   counted. Rationale: the pre-reg's stated intent is "the market's
   forward-looking vol forecast or the closest honest proxy"; the
   decomposition is that proxy, and beta = 1.0 was already the
   established fallback for uninformative beta estimates. The
   correction strengthens the baseline (it can only make GO harder).
   Both earlier evaluation runs are VOID and discarded; only the run
   reported here counts. Code: src/signal_lab/eventvol/iv_proxy.py
   (market_model_moments), merged as PR #59.
2. EVENTS PAST THE PRICE CALENDAR END: the pre-registration freezes
   prices at 2022-01-01..2026-07-15 and drops rows with partial label
   windows. Earnings events dated past the last trading day cannot form
   any label window; they are dropped with a logged count (not a silent
   filter): build_event_frame drops them before t0 mapping. No test or
   confirmation rows are affected (the calendar ends 2026-07-15). Code:
   src/signal_lab/eventvol/earnings.py, merged as PR #58.

## Notes on the secondary baselines

- har_vix (ridge on HAR+VIX features) is honestly bad on this frame:
  fold QLIKE 156.29 / 1.32 / 121.97 / 411.73 / 0.65. The damage comes
  from a handful of rows per fold where ridge extrapolates to
  near-zero forecasts on unusual event-day feature values (e.g. PLTR
  2022-11-07, PM 2024-10-22, UNH 2025-04-17); a single row with
  forecast 1e-4 against realized vol 0.03 contributes ~40,000 to the
  fold mean. The MCS eliminates har_vix first (p=0.0256). This is a
  genuine model failure on event rows, not a code bug: ridge was fit
  correctly per fold with frozen hyperparameters.
- naive (trailing realized_vol_5d) scores QLIKE 2.05-9.24: post-earnings
  realized vol is systematically above trailing vol, so persistence
  underforecasts. Eliminated by the MCS (p=0.0202).

## Kill criterion acknowledgment

Per the frozen kill criterion (docs/EVENTVOL_PREREGISTRATION.md): the
primary comparison is NO-GO, so the EVENTVOL campaign is DEAD. It is
not revived by trying other models, other features, other horizons,
other metrics, or other event definitions. No test-period evaluation
was run (the single authorized test eval is GO-gated, and there are 0
labelable test rows in any case). No confirmation data was touched for
any purpose. Any follow-up is a SEPARATELY pre-registered campaign;
nothing about EVENTVOL's data, features, or tuning carries a gate
claim into it.
