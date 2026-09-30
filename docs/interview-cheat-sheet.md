# Interview Cheat Sheet

Sixty-second pitch, then the questions this project is built to answer.

## The pitch

"I built an end-to-end news-sentiment market event engine. It ingests GDELT
news and market prices into DuckDB, resolves company mentions, trains a
classifier on imbalanced weak labels, runs a classical event study, validates
signals with a walk-forward backtest, and explains detected events using
cited retrieval over SEC filings. The first runs did not produce
statistically significant alpha, and I reported that honestly instead of
tuning the backtest until it looked profitable."

For 0.2.0, add: "We added Loughran-McDonald finance-lexicon features, then
proved our own evaluation was underpowered instead of tuning on noise, so we
scaled the data 5x with a full-year backfill."

## Questions it answers

**"Tell us about analysing a large dataset."**
365-day backfill: 784 news articles + 2,571 daily price rows across 10
tickers in DuckDB. GDELT throttled us (HTTP 429s); we built idempotent
ingestion with exponential backoff and retried failed windows overnight.
Schema: normalized news table (url, title, body, published_at,
source_domain) and OHLCV prices.

**"How would you link entities across documents?"**
Stage 2 resolves company mentions to tickers with rules plus optional spaCy
NER. The honest version: rule-based linking is brittle (aliases, "Apple"
the fruit vs the company); a production version would use a proper
entity-linking service like LSEG's PermID.

**"Build a model on a highly imbalanced dataset."**
~10% positive rate. Logistic regression on TF-IDF + 8 Loughran-McDonald
lexicon features (counts, rates, polarity, subjectivity). Compared plain,
class-weighted, resampled, threshold-tuned, and LightGBM candidates.
Selected by PR-AUC, never accuracy (a predict-nothing model scores 90%
accuracy). Weak labels from the market: top-decile next-day abnormal return
vs SPY.

**"How will you incorporate Statistics into Data Science?"**
Classical event study in stage 4: cumulative abnormal returns around
sentiment spikes with significance tests. Found 1 event (AAPL 2026-09-09,
CAR 3.7%) that was not statistically significant. Reported, not hidden.

**"How do you validate a model without fooling yourself?"**
Walk-forward backtest, sentiment strategy vs buy-and-hold, Sharpe 0 vs
0.383. No lookahead bias: timestamps sacred, features timestamped at or
before article time. When the 68-row evaluation proved too small for any
model comparison (temporal CV folds with zero positives), we said so and
scaled the data instead of tuning on noise. 0.3.0 adds purged walk-forward
CV with embargo and confidence intervals.

**"What would you do with more time?"**
The 0.3.0 roadmap: label quality first (multi-day windows, attention
filtering, a hand-labeled gold set to measure label noise), FinBERT
embeddings, market-context features (volatility, momentum, sector),
calibration with abstention, proper validation. Data scale is the binding
constraint, not features.

## Numbers to quote (2026-09-30, final backfill + retry)

- 1,709 news rows, 2,571 price rows, 10 tickers, 2025-10-15 to 2026-09-30.
  (Retry filled 44 of 69 empty GDELT windows; 25 remain empty on 429s.)
- 902 labeled examples (631 train / 271 test), ~10% positive.
- Classifier PR-AUC: challenger 0.131 vs baseline 0.126 vs 0.103 chance.
  Challenger F1 = 0.107 at threshold 0.5. Margin over baseline is 0.005:
  not a defensible win.
- Event study: 1 event (TSLA 2026-07-08), abnormal return -3.1%,
  not significant.
- Backtest (250 days, 5 bps costs): sentiment +1.1% vs buy-hold +14.5%,
  Sharpe 0.79 vs 1.10, paired t-test p = 0.31, not significant.
- Lexicon: 2,355 negative / 354 positive / 297 uncertainty words,
  Loughran & McDonald (2011).

## Lines to avoid

- Never claim the model is profitable or production-ready.
- Never claim the lexicon "improved" the model (effect inconclusive).
- Never present accuracy without the positive rate next to it.
- The honest story IS the differentiator. Don't sand it down.
