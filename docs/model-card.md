# Model Card: Signal Lab Sentiment Classifier

## Model details

- **What it is:** binary classifier predicting whether a news article about a
  company will coincide with an unusually large next-day stock move.
- **Architecture:** L2-regularized logistic regression on sparse text features
  plus dense finance-lexicon features, combined with `scipy.sparse.hstack`.
  One shared `featurize()` is used by training, prediction, and artifact
  scoring, so all three always agree.
- **Candidates compared:** plain logistic regression, class-weighted,
  resampled, threshold-tuned, and class-weighted LightGBM. The challenger is
  picked by PR-AUC on a time-based holdout, never by accuracy.
- **Version:** 0.2.0 adds Loughran-McDonald lexicon features (PR #2).
  0.1.0 was TF-IDF only.

## Training data

- **Source:** GDELT 2.1 DOC API news + yfinance daily prices, joined on
  ticker and date in DuckDB.
- **Labels (weak):** see "Labeling (0.3.0 workstream 1)" below. The 0.2.0
  scheme (label 1 if the ticker's next-trading-day abnormal return vs SPY
  lands in the top decile across the sample, else 0) is retained as the
  `--label-scheme baseline` option so old vs new stays measurable. Nobody
  labeled articles by hand; the market did the labeling. Positive rate ~10%.
- **Scale:** 0.1.0 trained on 68 labeled rows (~7 positives) from 30 days.
  0.2.0 trains on 337 labeled rows (235 train / 102 test) from a 365-day
  backfill across 10 tickers. Numbers below are pre-retry (2026-09-30); the
  backfill was still filling GDELT windows.

## Labeling (0.3.0 workstream 1)

The weak labels were the noisiest part of the system: a single next-day
return mixes real news-driven moves with random jumps. 0.3.0 replaces the
labeling with a configurable pipeline (`signal_lab.models.labels`):

- **Multi-day window (default):** label 1 if the ticker's *cumulative*
  abnormal return vs the market benchmark over trading days t+1..t+3 lands
  in the top decile, else 0. News takes time to digest; the window is
  configurable via `--window-days`.
- **Attention filter (default on):** an article is only labeled when its
  ticker-day carried enough news presence: at least `--min-articles` (2)
  distinct articles, OR a day strictly busier than 75% of that ticker's own
  daily counts (`--no-top-quartile` disables the second branch,
  `--no-attention` disables the filter). Quiet-day labels are mostly noise.
- **Market leg:** ticker return minus benchmark (default SPY) return over
  the same forward window; when the benchmark has no price on a date, the
  universe equal-weight mean fills in. Rows without a full forward window
  are dropped and counted.
- **Baseline retained:** `--label-scheme baseline` reproduces the 0.2.0
  t+1 top-decile labeling exactly (verified by test:
  `test_baseline_reproduces_legacy_t1_labeling`).
- **Provenance:** every stage-3 run emits a `label_config` block in its
  JSON, e.g.
  `{"scheme": "windowed", "window_days": 3, "quantile": 0.9,
  "benchmark": "SPY", "attention": {"enabled": true, "min_articles": 2,
  "top_quartile": true, "dropped_rows": 114, "dropped_pct": 0.1423},
  "cutoff": 0.040032, "positive_rate": 0.1033, "n_labeled": 687,
  "partial_window_rows_dropped": 91}`.
  Positive rate is expected to stay in the 5-15% band; outside it, the
  labeling (not the model) is the first suspect.
- **Gold set:** `python -m signal_lab.models.gold_set sample` draws a
  stratified sample of articles into a JSON file for hand-labeling ("does
  this read positive/negative for the company?"); `gold_set agree` reports
  the weak-label agreement rate, confusion breakdown, and Cohen's kappa
  once the human labels are filled in. The human labeling itself is still
  pending; when it lands, the agreement rate goes here.

## Features

- **TF-IDF** over article title + body (unigrams/bigrams).
- **Loughran-McDonald (8 dense features):** counts and per-token rates of
  positive, negative, and uncertainty words from the finance-specific
  dictionary (2,355 negative / 354 positive / 297 uncertainty words bundled
  in `models/data/lm_lexicon.json`), plus polarity
  `(pos-neg)/(pos+neg+1)` and subjectivity `(pos+neg)/n_tokens`.
- General-purpose lexicons misfire on finance text ("liability", "tax",
  "cost" are not negative in 10-Ks); LM was built from 10-K filings and
  encodes that domain knowledge.

## Evaluation (honest)

Metric of record is **PR-AUC** (area under the precision-recall curve):
the right metric when only 1 in 10 examples is positive. Accuracy is
reported but never used for selection (a "predict nothing matters" model
scores 90% accuracy and learns nothing).

| Setup | Baseline PR-AUC | Challenger PR-AUC |
|-------|----------------|-------------------|
| 0.1.0, 68 rows, TF-IDF only | 0.4167 | 0.2000 |
| 0.1.0 + lexicon, 68 rows | 0.1024 | 0.1429 |
| 0.2.0, 337 rows, TF-IDF + lexicon | 0.1626 | 0.0878 |

Chance level is ~0.10 (the positive rate). Read the table carefully:

- On 68 rows, adding features *moved PR-AUC from 0.42 to 0.10*. That is not
  "features hurt"; it is noise. With ~7 positives, a single holdout split
  cannot distinguish signal from luck. Temporal CV folds contained zero
  positives.
- On 337 rows the model scores 0.16 vs 0.10 chance. Barely above random.
  Both baseline and challenger have F1 = 0 at threshold 0.5.

**Conclusion we stand behind:** the small dataset cannot support trustworthy
model selection, and the lexicon's effect is inconclusive until the data is
bigger. Data scale, not features, is the binding constraint. We did not tune
the model until it looked good; we reported the bad numbers and scaled the
data instead.

## Intended use

Research and interview demonstration of an end-to-end news-sentiment
pipeline with honest evaluation. Not investment advice; the backtest exists
to check the signal, and so far the signal does not survive.

## Limitations

- Weak labels are noisy: a top-decile next-day jump mixes real news-driven
  moves with random volatility. 0.3.0 widens the window and filters
  quiet days, but the labels are still market-derived, not human.
- No market-context features yet (volatility regime, momentum, sector moves).
- No probability calibration; no abstention on low-confidence articles.
- Single time-based split, no confidence intervals yet.
- Gold-set human labels are still pending, so the weak-label agreement
  rate is unmeasured.
- See [docs/limitations.md](limitations.md) and the 0.3.0 roadmap
  ([docs/roadmap-0.3.0.md](roadmap-0.3.0.md)) for the planned fixes.

## Citation

Finance word lists derived from the Loughran-McDonald Master Dictionary:
Loughran, T. and McDonald, B. (2011). "When Is a Liability Not a Liability?
Textual Analysis, Dictionaries, and 10-Ks." *Journal of Finance* 66(1),
35-65. Used under the SRAF academic-research terms; commercial use requires
a license from the authors.
