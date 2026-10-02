# SYNTHESIS BETS: novel theoretical bets for the market-prediction program

**Status:** RESEARCH MEMO. No experiments were run for this document and
none are authorized by it. Each bet below is labeled SPECULATIVE: the
mechanism is reasoned from first principles or from an unpublished
combination of published parts, and the bet lives or dies by its own
pre-registered campaign. A bet that cannot be stated falsifiably was
dropped; the dropped list with reasons is at the end.

**Context.** Two campaigns are dead (direction: docs/NEGATIVE_RESULT.md;
volatility: docs/VOL_NEGATIVE_RESULT.md). The research sprint
(docs/RESEARCH_SPRINT.md) supplied three literature-backed candidates.
The user then asked for novel synthesis on top: new theory outside the
historically published scope, tested through the same honest machinery.
These bets are that synthesis. They combine in ways the surveyed
literature does not, and each carries a falsifiable prediction plus the
exact baseline that would kill it.

**Standing rules inherited.** Every bet that survives becomes its own
pre-registered campaign (docs/<NAME>_PREREGISTRATION.md committed before
any data is touched), with its own kill criterion and negative-result
doc on NO-GO. No campaign revives another. Frozen periods hold:
dev t0 <= 2026-06-30 for all work; test touched once only on GO;
confirmation never touched. yfinance only; GDELT DOC is hard-throttled
and is not used by any bet below (all features are price-derived).

---

## BET 1 (SPECULATIVE): cross-sectional dispersion predicts individual realized volatility (DISPVOL)

**Mechanism.** Disagreement models (Harris and Raviv 1993; Banerjee and
Kremer) predict that wider belief dispersion produces more trading and
larger price moves: dispersion today should forecast volatility
tomorrow. The return-prediction literature uses dispersion the other
way (Diether, Malloy, and Scherbina 2002: high dispersion predicts
*lower* subsequent returns); using dispersion as a *volatility*
predictor is the unpublished twist. Cross-sectional dispersion (the
daily cross-sectional standard deviation of returns) is observable
point-in-time from prices alone, needs no text, and is orthogonal to
the HAR lags by construction: it measures the market's cross-section,
not the stock's history.

**Falsifiable prediction.** Adding dispersion features to a HAR+VIX
model reduces out-of-sample QLIKE on 5-day realized volatility, with
the paired improvement significant at 95%.

**Baseline and why it is the right bar.** HAR+VIX is the literature's
best history-based arm (Blair-Poon-Taylor 2001; Busch-Christensen-
Nielsen 2011) and the arm the research sprint says we should have
tested first. The claim under test is narrowly "dispersion adds to the
best history-based model," so HAR+VIX is the bar. GARCH(1,1) is
reported as a secondary baseline (the Hansen-Lunde null).

**Metric.** QLIKE primary (Patton 2011); Diebold-Mariano and Model
Confidence Set for the arm comparison.

**Features (all point-in-time, yfinance only).** Daily cross-sectional
std of S&P 100 returns; 5-day change in dispersion; dispersion z-score
vs trailing 60 trading days; each also interacted with the stock's own
RV daily leg. Plus the frozen HAR+VIX feature set from the VOL
campaign.

**Pre-registration sketch.**
- Target: realized_vol_5d, same definition and frame as the VOL
  campaign (ticker-day, dev t0 <= 2026-06-30, 5-fold expanding purged
  walk-forward, 5-day embargo, seed 7).
- Primary comparison: HAR+VIX+dispersion vs HAR+VIX, paired per fold
  on QLIKE.
- GO iff mean paired QLIKE differential (baseline minus challenger) >
  0 AND the 95% two-sided t CI lower bound > 0.
- Kill criterion: miss on the primary comparison kills the campaign;
  write docs/DISPVOL_NEGATIVE_RESULT.md; no revival, no target switch.

**Why it might fail (stated upfront).** Dispersion may be a coincident
rather than leading indicator (it spikes *during* volatile regimes,
which HAR already captures); the cross-section of 100 large caps may
carry too little idiosyncratic disagreement to matter; any edge may
live at horizons shorter than 5 days.

---

## BET 2 (SPECULATIVE): attribution structure predicts post-event volatility (ATTRVOL)

**Mechanism.** The explanatory workbench decomposes "why did it move"
into drivers. First-principles claim: *how* a move happened predicts
what volatility does next. An event-day return concentrated in one
idiosyncratic driver (single-firm news) resolves uncertainty fast, so
volatility should decay quickly (the Ederington-Lee 1996 event-vol
cycle). An event-day return that is diffuse across market and sector
drivers, or that resembles past noisy episodes, should see volatility
persist. Two concrete, price-computable proxies: (1) driver
concentration, the Herfindahl index over a CAPM decomposition of the
event-day return into market, sector, and idiosyncratic legs; (2)
analogue-implied volatility, the mean post-event 5-day realized vol of
the k=10 nearest historical analogues by price-path similarity. The
unpublished combination is case-based reasoning (analogues) plus
attribution structure as *features* for a vol model, evaluated only on
event days where the literature says vol dynamics are most structured.

**Falsifiable prediction.** On event days (top-decile |abn_ret| per
ticker, quantile from trailing data only), HAR+VIX+attribution
features beat HAR+VIX on QLIKE for 5-day post-event realized vol, with
the paired improvement significant at 95%.

**Baseline and why it is the right bar.** HAR+VIX again: the claim is
"attribution structure adds to the best history-based model on the
days when structure should matter most." Restricting evaluation to
event days is pre-registered (not cherry-picked after the fact);
full-sample performance is reported as a secondary.

**Metric.** QLIKE primary; DM/MCS for the comparison.

**Features (all point-in-time, yfinance only; no GDELT).** Driver
Herfindahl from the market/sector/idiosyncratic decomposition of the
event-day return; idiosyncratic share of |abn_ret|; analogue-implied
vol (k=10, price-path cosine similarity on trailing 20-day return
paths, analogues drawn from train history only); signed event return
interacted with concentration. Plus the frozen HAR+VIX set.

**Pre-registration sketch.**
- Target: realized_vol_5d on event-day rows only (event = trailing
  top-decile |abn_ret|; definition frozen before data is touched).
- Primary comparison: HAR+VIX+attribution vs HAR+VIX, paired per fold
  on QLIKE over event-day test rows.
- GO iff mean paired differential > 0 AND 95% CI lower bound > 0.
- Kill criterion: miss kills the campaign;
  docs/ATTRVOL_NEGATIVE_RESULT.md; no revival.

**Why it might fail (stated upfront).** The CAPM decomposition may be
too coarse to capture true driver structure; analogues by price path
may match noise, not mechanism; event-day vol may already be saturated
by the event-day return magnitude itself (which HAR includes), leaving
nothing for structure to add.

---

## BET 3 (SPECULATIVE): conformal-abstention-gated event-magnitude prediction (CONFMAG)

**Mechanism.** Magnitude may be predictable only in a subset of
regimes, and the model itself can detect those regimes: train a
regressor for |excess_3d| (absolute 3-day abnormal return), wrap it in
the split-conformal machinery from the extraction program (MAPIE,
alpha=0.2), and predict *only* on rows where the conformal interval is
tight (width below the 25th percentile of calibration widths). The
interval width is a learned regime detector: it is small where the
model's residuals are systematically small, i.e. where the feature
regime is informative. This is selective prediction ("regression with
a reject option," cf. Geifman and El-Yaniv) applied to event magnitude
in finance, which the surveyed literature does not do. The evaluation
compares model vs baseline *on the same selected subset*, so the
selection cannot manufacture an edge by itself.

**Falsifiable prediction.** On the pre-registered selected subset,
the gated model beats a constant forecast of |excess_3d| on MSE with
the paired improvement significant at 95%, with selection rate between
10% and 40% of test rows (guards against degenerate empty or
near-universal selection).

**Baseline and why it is the right bar.** The constant forecast (train
mean |excess_3d|) is the honest bar the research sprint names for
magnitude targets (memo section 2.3: "a constant forecast" is the
documented baseline; no published strong baseline exists). A
GARCH-implied magnitude (sigma_3d * sqrt(2/pi)) is reported as a
secondary baseline.

**Metric.** MSE on |excess_3d| primary; MAE secondary; selection rate
and interval coverage reported.

**Pre-registration sketch.**
- Target: |excess_3d| per (ticker, day); model: LightGBM regressor on
  HAR-style + event features; split-conformal intervals calibrated on
  a held-out calibration split inside dev.
- Selection rule frozen before evaluation: predict on test rows with
  interval width <= 25th percentile of calibration widths.
- Primary comparison: gated model vs constant baseline, paired per
  fold on MSE over selected test rows.
- GO iff mean paired MSE differential (baseline minus model) > 0 AND
  95% CI lower bound > 0 AND selection rate in [0.10, 0.40].
- Kill criterion: miss on any conjunct kills the campaign;
  docs/CONFMAG_NEGATIVE_RESULT.md; no revival.

**Why it might fail (stated upfront).** Conformal width may track
feature-space density rather than predictability (tight intervals
where data is dense, which need not be where magnitude is forecastable);
the selected subset may be too small for the CI to clear zero; the
constant baseline may already be near-optimal for absolute returns,
which are notoriously close to unpredictable.

---

## Recommended campaign order for the synthesis bets

DISPVOL, then ATTRVOL, then CONFMAG. Rationale: kill the cheapest
falsifiable hypotheses first. DISPVOL reuses the VOL harness almost
unchanged (new features only). ATTRVOL needs new feature engineering
(decomposition, analogues) but the same harness. CONFMAG is the most
complex (conformal regression machinery, new target, selection
mechanics) and the most novel; it runs last so its complexity is only
paid if the cheaper bets die.

## Bets considered and dropped

- **VPIN / order-flow toxicity from daily data.** Dropped: VPIN
  (Easley, Lopez de Prado, O'Hara) requires volume-bucketed intraday
  bars; it cannot be constructed from daily OHLCV without fabricating
  the bucketing the estimator needs. A daily proxy would test the
  proxy, not the theory.
- **News-surprise magnitude via GDELT.** Dropped: the GDELT DOC API is
  hard-throttled on this IP (gate-2 and VOL both hit 429s), and text
  arms have already failed twice. A bet that cannot get data is not a
  bet.
- **LLM reasoning over filings for event prediction.** Dropped: needs
  a filing-text pipeline we do not have, inference costs are nonzero
  (the program is no-cost), and no published baseline shows
  filing-reasoning beating price-based event features. Speculation
  without a feasible test.
- **Overnight/intraday volatility decomposition as a standalone bet.**
  Folded into ATTRVOL's feature set instead: the close-to-open vs
  open-to-close split is computable from daily OHLC and belongs with
  attribution structure, not as its own campaign.

## Feasibility flag for Candidate A (event-window vol vs implied vol)

The research sprint's Candidate A names "pre-event ATM implied vol" as
a baseline. Historical per-stock implied volatility is NOT available
from yfinance (option chains are current-only). Before that campaign's
pre-registration is committed, its coordinator must resolve one of:
(a) restrict the IV baseline to the index (SPY vs VIX, where history
exists); (b) find a free historical IV source and verify it; or
(c) reformulate the baseline honestly (e.g., VIX term-structure
features) and document the change. Committing a pre-registration
against an unobtainable baseline is forbidden.
