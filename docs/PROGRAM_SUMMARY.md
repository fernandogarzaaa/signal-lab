# Signal Lab market-prediction program: final summary

**Status:** PROGRAM EXHAUSTED. All 8 pre-registered campaigns reached
NO-GO verdicts under their frozen decision rules. No viable market
prediction engine was found. This document is the complete program
record. No further campaign may start without the user's explicit new
direction.

## Program history

The program ran in three stages:

1. **Direction program (gate 1 + blocked gate 2):** news-sentiment
   direction prediction, 2026-09-30..2026-10-01. Gate 1 NO-GO on 693
   dev rows (+0.0035 PR-AUC, 95% CI [-0.0091, 0.0160], p=0.48). Gate 2
   blocked by GDELT DOC API hard-throttling; fallback executed.
   See `docs/NEGATIVE_RESULT.md`.
2. **Extraction program (APP-000..APP-006, 2026-10-01..2026-10-02):**
   product improvements to the explanatory workbench (conformal
   abstention, event-study significance tests, TabPFN benchmark,
   leakage assertions, gdelt-client migration, purgedcv cross-check,
   Deflated Sharpe Ratio). Not a prediction program.
3. **Volatility/magnitude campaigns (2026-10-02..2026-10-03):** seven
   sequentially pre-registered campaigns, each with a frozen success
   rule and kill criterion, each dead on NO-GO. Preceded by
   `docs/RESEARCH_SPRINT.md` (literature + OSS reality check) and
   `docs/SYNTHESIS_BETS.md` (three speculative bets).

## Campaign record

Every campaign below used: frozen dev period t0 <= 2026-06-30;
purged + embargoed expanding walk-forward (seed 7, purge j==k);
frozen universe S&P 100 (101) + SPY = 102 tickers in
`data/universe_vol.csv`; yfinance daily prices 2022-01-01..2026-07-15;
no GDELT anywhere. Test (2026-07-01..08-31) and confirmation
(t0 >= 2026-09-01) were never touched in any campaign.

| # | Campaign | Target | Baseline / arms | Metric | Verdict (numbers) | Pre-reg | Results |
|---|---|---|---|---|---|---|---|
| 1 | Direction (gate 1) | weak label: 3d abnormal return direction | logreg_balanced price-only vs price+TF-IDF | PR-AUC diff | NO-GO: +0.0035, 95% CI [-0.0091, 0.0160], p=0.48 | docs/GATE.md | docs/NEGATIVE_RESULT.md |
| 2 | VOL | realized_vol_5d | GARCH(1,1) vs LightGBM HAR | QLIKE diff | NO-GO: mean +0.0115, 95% CI [-0.0966, +0.1196] | docs/VOL_PREREGISTRATION.md | docs/VOL_NEGATIVE_RESULT.md |
| 3 | EVENTVOL | realized_vol_5d on earnings rows | IV proxy vs LightGBM HAR+VIX+event | QLIKE diff | NO-GO: mean -0.2487, 95% CI [-1.3025, +0.8052] | docs/EVENTVOL_PREREGISTRATION.md | docs/EVENTVOL_NEGATIVE_RESULT.md |
| 4 | HARVIX | realized_vol_5d | GARCH(1,1) vs ridge HAR+VIX | QLIKE diff | NO-GO: mean -0.8523, 95% CI [-3.1085, +1.4039] | docs/HARVIX_PREREGISTRATION.md | docs/HARVIX_NEGATIVE_RESULT.md |
| 5 | TAILQ | 5% quantile of 3d event return | CAViaR vs LightGBM quantile | pinball loss diff | NO-GO (rule not cleared) | docs/TAILQ_PREREGISTRATION.md | docs/TAILQ_NEGATIVE_RESULT.md |
| 6 | DISPVOL | realized_vol_5d | ridge HAR+VIX vs +dispersion | QLIKE diff | NO-GO: mean -0.5458, 95% CI [-2.0802, +0.9886] | docs/DISPVOL_PREREGISTRATION.md | docs/DISPVOL_NEGATIVE_RESULT.md |
| 7 | ATTRVOL | realized_vol_5d on event rows | ridge HAR+VIX vs +attribution | QLIKE diff | NO-GO: mean -0.1428, 95% CI [-0.3625, +0.0770] | docs/ATTRVOL_PREREGISTRATION.md | docs/ATTRVOL_NEGATIVE_RESULT.md |
| 8 | CONFMAG | \|excess_3d\|, CQR-gated | constant vs LightGBM gated | MSE diff (selected) | NO-GO: mean +0.000011, 95% CI [-0.000011, +0.000034]; sel rate 0.2305 in guard | docs/CONFMAG_PREREGISTRATION.md | docs/CONFMAG_NEGATIVE_RESULT.md |

Experiment registry: `docs/EXPERIMENT_REGISTRY.md` (41 tested
experiments). The research sprint and synthesis bets that designed
campaigns 2-8: `docs/RESEARCH_SPRINT.md`, `docs/SYNTHESIS_BETS.md`.

## The honest bottom line

Under pre-registered rules, with honest baselines, purged
walk-forward validation, and no peeking: **nothing tested beats the
simple baselines.** GARCH(1,1) held against LightGBM on realized
volatility; the constant train-mean held against conformal-gated
LightGBM on event magnitude; implied-vol proxies held on earnings
events; CAViaR held on tail quantiles; and two speculative feature
families (cross-sectional dispersion, attribution structure) made
forecasts no better, sometimes worse. This matches the published
literature surveyed in `docs/RESEARCH_SPRINT.md` (Hansen-Lunde 2005:
330 models, none beats GARCH(1,1) on daily data; sentiment alphas
decayed to zero years ago). The negative results are genuine, not
metric artifacts (QLIKE per Patton 2011; pinball per Engle-Manganelli
2004).

## Where an edge might still exist (evidence, not promises)

From the research sprint and the campaign results:

1. **Intraday realized-volatility features.** The literature is clear
   that HAR's edge comes from high-frequency realized measures we do
   not have (daily bars only). A campaign built on intraday data
   (e.g., 5-minute bars) tests the one input the literature says
   matters. Data cost and engineering effort are the barriers, not
   theory.
2. **Options-market microstructure around events.** Candidate A
   (EVENTVOL) used an index-IV proxy because per-stock historical IV
   was unobtainable from yfinance. The literature's highest-conviction
   claim is that implied volatility subsumes history-based models;
   the campaign we ran could not test per-stock IV directly. With
   options data, the honest question (does anything add to per-stock
   IV?) is still open.
3. **Cross-sectional ML with realistic frictions.** Gu-Kelly-Xiu-style
   monthly cross-sectional prediction shows real out-of-sample R2,
   but the edge concentrates in microcaps where transaction costs
   erase it. A campaign with explicit cost modeling and
   liquidity-filtered universes would test whether anything survives
   costs. The memo flags this as real-but-fragile.
4. **Methodology as the product.** The program's most defensible
   output is not a predictor but the validation machinery itself:
   pre-registered campaigns, purged walk-forward with embargo, honest
   baselines, fail-loud data gates, and published negative results.
   The OSS survey found this discipline almost nowhere in public
   market-prediction projects.

## Program rules honored to the end

- Every campaign was pre-registered before any data was touched;
  every verdict applied the frozen rule exactly; every deviation is
  logged in its result document.
- Frozen periods were never violated: test and confirmation data
  were never read, scored, or inspected in any campaign.
- All 41 tested experiments are logged in
  `docs/EXPERIMENT_REGISTRY.md`. All negative results are committed
  and public.
