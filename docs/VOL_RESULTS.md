# VOL campaign results: volatility prediction (working document)

Pre-registration: `docs/VOL_PREREGISTRATION.md` (frozen, binding). This document is the working record; the committed results PR carries the final version.

## Configuration

- Champion (from single purged selection split): **lightgbm**
- Baseline: GARCH(1,1) MLE, trailing 252 trading days, analytic 5-day term structure; fit failures fall back to naive persistence
- Sanity arm: naive (trailing realized_vol_5d)
- Validation: 5-fold expanding purged walk-forward on t0, seed 7, min_train 50, purge j==k, 5-trading-day embargo
- Primary metric: QLIKE on variance (lower is better)
- Dev rows: 113511 (111267 after NaN-feature drop)
- GARCH naive fallbacks on test rows: 5166 / 92667

## Per-fold QLIKE (variance)

| fold | test range | n_train | n_test | purged | embargoed | QLIKE champion | QLIKE garch11 | QLIKE naive | d = garch - champ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-10-31..2023-07-27 | 18100 | 18500 | 500 | 0 | 0.546528 | 0.696149 | 1.599077 | +0.149621 |
| 2 | 2023-07-28..2024-04-23 | 36600 | 18600 | 500 | 0 | 0.613764 | 0.600372 | 2.041675 | -0.013392 |
| 3 | 2024-04-24..2025-01-15 | 54700 | 18480 | 500 | 500 | 0.685183 | 0.657060 | 2.172974 | -0.028122 |
| 4 | 2025-01-16..2025-10-08 | 72673 | 18625 | 505 | 1002 | 0.754244 | 0.785199 | 2.043785 | +0.030955 |
| 5 | 2025-10-09..2026-06-30 | 90788 | 18462 | 510 | 1507 | 0.600885 | 0.519340 | 1.759039 | -0.081545 |

## Verdict computation (frozen decision rule)

- d_k = QLIKE(garch11) - QLIKE(champion) per fold: +0.149621, -0.013392, -0.028122, +0.030955, -0.081545
- mean(d) = +0.011503, se = 0.038925, df = 4.0, t(0.975) = 2.7764
- 95% two-sided Student-t CI on mean(d): [-0.096569, +0.119576]
- Diebold-Mariano on fold-level QLIKE differentials (paired-t form): stat = +0.2955, two-sided p = 0.7823, n = 5.0
- GO iff mean(d) > 0 AND CI lower bound > 0: **NO-GO**

## Secondary metrics (per fold, no gate authority)

| fold | MSE-var champion | MSE-var garch11 | MAE-vol champion | MAE-vol garch11 |
| --- | --- | --- | --- | --- |
| 1 | 0.00000037 | 0.00000047 | 0.007913 | 0.007853 |
| 2 | 0.00000048 | 0.00000051 | 0.005822 | 0.006524 |
| 3 | 0.00000056 | 0.00000085 | 0.006702 | 0.007146 |
| 4 | 0.00000139 | 0.00000167 | 0.007625 | 0.009273 |
| 5 | 0.00000106 | 0.00000112 | 0.007491 | 0.007927 |

## Selection honesty (DSR, reported, not a gate)

per-row QLIKE gains over naive as pseudo-returns, freq=1; reported only, not a gate

- lightgbm: Sharpe = +0.1466, DSR = 1.0000, likely_false_discovery = False
- ridge: Sharpe = +0.1433, DSR = 1.0000, likely_false_discovery = False

## Deviations

- TEXT ARMS SKIPPED (Phase 4, exploratory): the GDELT DOC API probe (query '("Apple Inc. stock" OR "AAPL stock") sourcelang:english' for 2026-06-01, run 2026-10-02T11:05:20Z) returned 0 articles without error, but the first substantive quarterly query ('("AAPL stock") sourcelang:english', 2026-04-01..2026-06-30, attempted 2026-10-02~11:07Z) got HTTP 429 Too Many Requests on all 4 attempts (client backoff 15s/120s/120s) and raised gdelt_client.errors.RateLimitError - the same hard throttle as gate-2. No proxies or rate-limit evasion attempted. The TF-IDF + Loughran-McDonald text arms are not run in this campaign.
- BASELINE SCALE (deviation from the pre-registration's literal text): docs/VOL_PREREGISTRATION.md writes the GARCH(1,1) forecast as the 5-day term-structure variances 'summed over h=1..5, square-rooted'. Taken literally that is sqrt(sum of 5 daily variances) = 5-day cumulative volatility, sqrt(5)~2.24x larger than the frozen target realized_vol_5d, which is defined (docs/TARGETS_AND_FEATURES.md v2.0, units 'decimal, daily') as the sample std (ddof=1) of daily log returns over (t0,t0+5]. The literal reading puts the baseline on the wrong scale by construction (even a perfect-shape forecast scores QLIKE~0.81 on scale alone; observed GARCH QLIKE was 1.2-1.4 vs champion 0.55-0.75), which would make the champion-vs-baseline comparison meaningless. Implemented instead: sqrt(mean of the 5 h-step variances), i.e. the term-structure forecast of expected daily variance over the window, square-rooted to the target's daily scale - the same scale as the naive arm (trailing realized_vol_5d). The first evaluation run used the literal sqrt(sum); its GO verdict is VOID and discarded - only the scale-corrected run below counts.
