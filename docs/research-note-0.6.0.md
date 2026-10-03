# Research Note 0.6.0: Eight Campaigns, Eight Honest NO-GOs

## Summary

Between October 1 and October 3, 2026, Signal Lab ran a pre-registered
market-prediction program: eight sequential campaigns, each committed to
a frozen design before a single row of data was touched, each with a
pre-registered success rule and kill criterion. **All eight returned
NO-GO.** No viable prediction engine was found. This note reports the
full record, what the failures teach, and where the evidence says an
edge might still exist.

The program's discipline is the point. Every campaign used purged
walk-forward validation with embargo, frozen development/test/
confirmation periods (the test and confirmation periods were never
touched in any campaign), paired statistical tests with confidence
intervals, and a Deflated Sharpe Ratio honesty check on model
selection. Negative results are first-class outputs, written up in
`docs/*_NEGATIVE_RESULT.md`.

## The Eight Campaigns

| # | Campaign | Target vs baseline | Verdict |
|---|----------|-------------------|---------|
| 1 | Direction | News sentiment vs price-only, PR-AUC | NO-GO: diff +0.0035, 95% CI [-0.0091, 0.0160], p=0.48 |
| 2 | VOL | LightGBM on HAR features vs GARCH(1,1), QLIKE | NO-GO: mean diff +0.0115, CI [-0.0966, +0.1196] |
| 3 | EVENTVOL | Event-window vol vs implied vol | NO-GO: mean diff -0.2487 |
| 4 | HARVIX | HAR+VIX vs GARCH(1,1), QLIKE | NO-GO: mean diff -0.8523 |
| 5 | TAILQ | Tail-quantile VaR vs CAViaR, pinball loss | NO-GO: decision rule not cleared |
| 6 | DISPVOL | Cross-sectional dispersion vs HAR+VIX | NO-GO: mean diff -0.5458 |
| 7 | ATTRVOL | Attribution-driven features vs HAR+VIX | NO-GO: mean diff -0.1428 |
| 8 | CONFMAG | Conformal-gated magnitude vs constant, MSE | NO-GO: mean diff +0.000011, CI [-0.000011, +0.000034] |

Full per-campaign records: `docs/PROGRAM_SUMMARY.md` and the
`*_NEGATIVE_RESULT.md` files.

## What the Failures Teach

**GARCH(1,1) is a genuinely hard null on daily data.** This is not our
idiosyncratic finding. Hansen and Lunde (2005) compared 330 volatility
specifications and could not reject that none beats GARCH(1,1). Our
VOL loss is the literature's expected outcome of a fair fight, and our
QLIKE choice was the right metric for it (Patton 2011).

**HAR's edge requires intraday data we do not have.** The literature's
HAR-beats-GARCH results use precise intraday realized volatility as
input. Fed daily bars and a noisy 5-day sample-std target, the
persistence differences compress. Losing that shootout is within the
expected outcomes.

**Implied volatility subsumes history-based models.** Blair, Poon, and
Taylor (2001) show VIX gives the most accurate out-of-sample volatility
forecasts at all horizons, and history adds nothing once IV is in the
model. Our EVENTVOL campaign confirmed the harsh version: nothing we
tried added to the market's own forecast.

**News-sentiment direction edges decayed to zero.** Our gate-1 result
(prices plus sentiment indistinguishable from prices alone) sits
exactly where the literature says the edge now lives: Tetlock (2007)
found week-long effects; Ke, Kelly, and Xiu (2024) find full
incorporation within about four trading days. FinBERT-family models
are text classifiers with high-80s in-domain F1 and no documented
sentiment-to-return edge. Our failure matches the consensus.

**Drawdown prediction has no credible baseline.** The best public
numbers (near-perfect classifiers, Sharpe 2.5 timers) are preprints
or low-tier journals with implausible magnitudes. We did not attempt
it; the literature gives no honest bar to beat.

## Where the Evidence Says an Edge Might Still Exist

Four avenues survived the survey with their evidentiary basis intact.
None was attempted; each needs data or infrastructure beyond daily
yfinance bars:

1. **Intraday realized-vol features.** HAR's documented edge comes
   from intraday RV; with proper intraday data the HAR-VIX shootout
   could be re-run fairly.
2. **Per-stock options implied volatility.** Index-level IV subsumes
   history; the per-stock analog was never tested for lack of
   options data.
3. **Cost-modeled cross-sectional ML.** Gu, Kelly, and Xiu (2020)
   document real out-of-sample R2, but the economics live in
   microcaps where trading costs erase them. Needs CRSP-scale data
   and honest cost modeling.
4. **Methodology as product.** The field is saturated with hype
   repos and starved of honest validation. Purged walk-forward,
   pre-registration, multiple-testing correction, and published
   negative results are themselves a differentiator.

## What Signal Lab Is

Signal Lab remains what it was built to be: a cited market-event
attribution workbench that answers "why did this ticker move on this
date," with historical analogues and a news watchlist. It explains
past moves; it does not predict future ones. The prediction program
above is closed, and its honest ending is part of the record, not a
retraction of the product.
