# Limitations

What Signal Lab cannot do, stated plainly. Every limitation here is either
accepted, worked around, or scheduled on the 0.3.0 roadmap.

## Data

- **GDELT throttling.** The free news API rate-limits aggressively (HTTP 429).
  The 365-day backfill needed multiple passes with backoff; some windows may
  stay empty. Ingestion is idempotent, so retries are safe, but coverage is
  never guaranteed.
- **Weak labels are noisy.** "This article mattered" is defined as a
  top-decile next-day abnormal return. That mixes genuine news-driven moves
  with random jumps, earnings-day volatility, and market-wide moves the SPY
  adjustment didn't fully remove.
- **Single-day window.** News often takes days to digest; labeling on t+1
  alone punishes the model for slow market reactions.
- **Survivorship and selection bias.** The 10-ticker universe is large-cap US
  equities chosen by hand, not a representative sample. Results do not
  generalize to small caps, other markets, or other regimes.
- **No PySpark.** The pipeline uses DuckDB, not Spark. At this data scale
  that is the right tool; it is not the tool the LSEG posting names.

## Model

- **Barely above chance.** Final 0.2.0: challenger PR-AUC 0.131 vs 0.126
  unweighted baseline vs 0.103 chance on 902 rows. The honest reading is
  "not a usable signal yet," and the pipeline is built to say exactly that.
- **Small-sample evaluation.** 902 rows with ~10% positives cannot support
  trustworthy model selection. The challenger beats the baseline by 0.005
  PR-AUC: not a defensible win. Single holdout splits swing wildly.
- **No calibration.** Predicted probabilities are not calibrated; a "70%"
  prediction does not happen 70% of the time. No abstention mechanism for
  low-confidence articles.
- **Text-only.** The model reads the article but knows nothing about market
  context: volatility regime, momentum, sector moves, or whether the whole
  market jumped that day.
- **No deep learning.** Logistic regression and LightGBM on TF-IDF +
  lexicon features. FinBERT embeddings are scoped for 0.3.0.

## Evaluation

- **Single time-based split**, no confidence intervals, no purged
  walk-forward CV yet (scoped for 0.3.0).
- **Backtest costs are simple.** 5 bps one-way on turnover is modeled; no
  slippage or market impact. Any real deployment would look worse, not
  better. The sentiment strategy still trailed buy-and-hold (+1.1% vs
  +14.5%, paired p = 0.31, not significant).
- **One regime.** All data is 2025-2026 large-cap US equities. Nothing here
  says anything about how the approach behaves in a crisis, a rate shock,
  or a different market.

## Deployment

- **Research demo, not a product.** The npm package, dashboard, and seed
  snapshot exist to make the pipeline runnable and demoable, not to trade
  on. Nothing in this repo is investment advice.
- **Lexicon licensing.** The bundled Loughran-McDonald word lists are free
  for academic research; commercial use requires a license from the authors
  (loughranmcdonald@gmail.com). The repo's MIT license does not extend to
  that data. See the notice in `src/signal_lab/models/data/`.

## What we're doing about it

The 0.3.0 roadmap ([docs/roadmap-0.3.0.md](roadmap-0.3.0.md), tracking issue
#4) addresses the fixable items in priority order: label quality first
(multi-day windows, attention filtering, a hand-labeled gold set), then
FinBERT features, market-context features, calibration with abstention, and
purged walk-forward validation with confidence intervals.
