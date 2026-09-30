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
- **Labels (weak):** label 1 if the ticker's next-trading-day abnormal return
  vs SPY lands in the top decile across the sample, else 0. Nobody labeled
  articles by hand; the market did the labeling. Positive rate ~10%.
- **Scale:** 0.1.0 trained on 68 labeled rows (~7 positives) from 30 days.
  0.2.0 trains on 902 labeled rows (631 train / 271 test) from a 365-day
  backfill across 10 tickers: 1,709 GDELT articles and 2,571 daily price rows
  (2025-10-15 to 2026-09-30). The overnight retry filled 44 of 69 empty
  GDELT windows; 25 remain empty after HTTP 429 throttling.

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
| 0.2.0 partial, 337 rows, TF-IDF + lexicon | 0.1626 | 0.0878 |
| 0.2.0 final, 902 rows, TF-IDF + lexicon | 0.1260 | 0.1309 |

Chance level is ~0.10 (the positive rate). Read the table carefully:

- On 68 rows, adding features *moved PR-AUC from 0.42 to 0.10*. That is not
  "features hurt"; it is noise. With ~7 positives, a single holdout split
  cannot distinguish signal from luck. Temporal CV folds contained zero
  positives.
- On 337 partial rows the challenger scored below the baseline (0.088 vs
  0.163), again noise: both had F1 = 0 at threshold 0.5.
- On the final 902 rows the challenger (class-weighted LightGBM) scores
  0.131 vs 0.126 baseline vs 0.103 chance. Slightly above random, and the
  challenger finally has non-zero F1 (0.107 at threshold 0.5), but the
  margin over the unweighted baseline is 0.005 PR-AUC: not a defensible win.

**Conclusion we stand behind:** even at 902 rows the dataset cannot support
trustworthy model selection, and the lexicon's effect remains inconclusive.
Data scale and label quality, not features, are the binding constraints.
The walk-forward backtest agrees: the sentiment-momentum strategy returned
+1.1% vs +14.5% buy-and-hold over 250 days, Sharpe 0.79 vs 1.10, paired
t-test p = 0.31 (not significant). We did not tune the model until it
looked good; we reported the numbers as they are.

## Intended use

Research and interview demonstration of an end-to-end news-sentiment
pipeline with honest evaluation. Not investment advice; the backtest exists
to check the signal, and so far the signal does not survive.

## Limitations

- Weak labels are noisy: a top-decile next-day jump mixes real news-driven
  moves with random volatility.
- Single-day label window; news takes longer than a day to digest.
- No market-context features yet (volatility regime, momentum, sector moves).
- No probability calibration; no abstention on low-confidence articles.
- Single time-based split, no confidence intervals yet.
- See [docs/limitations.md](limitations.md) and the 0.3.0 roadmap
  ([docs/roadmap-0.3.0.md](roadmap-0.3.0.md)) for the planned fixes.

## Citation

Finance word lists derived from the Loughran-McDonald Master Dictionary:
Loughran, T. and McDonald, B. (2011). "When Is a Liability Not a Liability?
Textual Analysis, Dictionaries, and 10-Ks." *Journal of Finance* 66(1),
35-65. Used under the SRAF academic-research terms; commercial use requires
a license from the authors.
