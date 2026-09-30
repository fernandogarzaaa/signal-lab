# Model Card: Signal Lab Sentiment Classifier

## Model details

- **What it is:** binary classifier predicting whether a news article about a
  company will coincide with an unusually large next-day stock move.
- **Architecture:** L2-regularized logistic regression on sparse text features
  plus dense finance-lexicon features, combined with `scipy.sparse.hstack`.
  One shared `featurize()` is used by training, prediction, and artifact
  scoring, so all three always agree. 0.3.0 workstream 3 appends 7 dense
  market-context features (volatility regime, momentum, sector-relative
  strength, article metadata) through the same shared path via an optional
  `dense_extra` argument.
- **Candidates compared:** plain logistic regression, class-weighted,
  resampled, threshold-tuned, and class-weighted LightGBM. The challenger is
  picked by PR-AUC on a time-based holdout, never by accuracy.
- **Version:** 0.2.0 adds Loughran-McDonald lexicon features (PR #2).
  0.1.0 was TF-IDF only. 0.3.0 workstream 3 adds market-context features
  and a `feature_importance` block in the stage-3 JSON.

## Training data

- **Source:** GDELT 2.1 DOC API news + yfinance daily prices, joined on
  ticker and date in DuckDB.
- **Labels (weak):** see "Labeling (0.3.0 workstream 1)" below. The 0.2.0
  scheme (label 1 if the ticker's next-trading-day abnormal return vs SPY
  lands in the top decile across the sample, else 0) is retained as the
  `--label-scheme baseline` option so old vs new stays measurable. Nobody
  labeled articles by hand; the market did the labeling. Positive rate ~10%.
- **Scale:** 0.1.0 trained on 68 labeled rows (~7 positives) from 30 days.
  0.2.0 trains on 902 labeled rows (631 train / 271 test) from a 365-day
  backfill across 10 tickers: 1,709 GDELT articles and 2,571 daily price rows
  (2025-10-15 to 2026-09-30). The overnight retry filled 44 of 69 empty
  GDELT windows; 25 remain empty after HTTP 429 throttling.

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
- **Market context (0.3.0 workstream 3, 7 dense features,
  `signal_lab.models.context_features`):** every feature is knowable at
  prediction time; the leakage audit below pins this down.
  - `ctx_vol20_pct`: percentile of the trailing-20-trading-day realized
    volatility (sample std of daily returns) within its own trailing
    up-to-252-trading-day history (min 20 observations).
  - `ctx_mom5` / `ctx_mom20`: trailing 5- and 20-trading-day returns.
  - `ctx_sector_rel5`: ticker trailing-5d return minus the sector leg's
    trailing-5d return. Sector comes from a static ticker to GICS sector
    ETF map (AAPL/MSFT/NVDA to XLK, GOOGL/META to XLC, AMZN/TSLA to XLY,
    JPM to XLF, JNJ to XLV, XOM to XLE); sector ETF prices are not in the
    backfill, so the leg currently falls back to SPY and the feature equals
    the 5-day abnormal return vs SPY. The interface accepts real ETF price
    series so a future backfill can swap the leg without changing callers.
  - `ctx_src_tier`: source-domain tier, 2 = wire/top financial outlet,
    1 = established press, 0 = other/unknown (heuristic outlet map, suffix
    matched so subdomains inherit their parent tier).
  - `ctx_word_count`: word count of title + body snippet.
  - `ctx_hour_bucket`: UTC publish hour in quarters (0 = 00-05, 1 = 06-11,
    2 = 12-17, 3 = 18-23).

## Leakage audit (0.3.0 workstream 3)

Claim: no market-context feature uses information from after the
article's publish time. The guarantee is structural, not just tested:
for an article published on calendar date D, the as-of trading day is the
latest trading day *strictly before* D, and every rolling window
(5d momentum, 20d momentum, 20d vol, up-to-252d vol history) ends on or
before that day. Metadata features come from the article row itself.
`tests/test_context_features.py::test_no_leakage_post_article_spike`
constructs a synthetic series, computes features, then applies a 3x price
spike *after* the article date and asserts every price feature is
unchanged; a second spike *before* the as-of day must move the features,
proving the test is not vacuous. A companion test asserts an article
published intraday on a trading day does not see that day's close.

## Feature importance (0.3.0 workstream 3)

Every stage-3 run emits a `feature_importance` block for the persisted
(best) model: `|coefficient| x feature-std` for linear models (so raw
scale differences between TF-IDF, lexicon, and context features do not
dominate), `feature_importances_` for tree models. It reports the top 20
text terms plus all 15 dense (8 lexicon + 7 context) features, ranked.

Observed ranking on the seed snapshot (logreg_plain, 56 labeled rows /
39 train / 17 test; method `abs_coef_times_std`):

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | ctx_word_count | 0.939 |
| 2 | lm_neg_count | 0.302 |
| 3 | lm_pos_count | 0.224 |
| 4 | lm_unc_count | 0.212 |
| 5 | ctx_src_tier | 0.119 |
| 6 | lm_polarity | 0.095 |
| 7 | ctx_hour_bucket | 0.044 |
| 8 | ctx_vol20_pct | 0.041 |
| 9 | ctx_mom5 | 0.009 |
| 10 | ctx_sector_rel5 | 0.008 |
| 11 | ctx_mom20 | 0.003 |
| 12-15 | lm_*_rate, lm_subjectivity | ~0.000 |

Read this table as a JSON-shape check, not as insight: with 39 training
rows the ranking is noise, and `ctx_word_count` on top almost certainly
reflects a spurious length correlation in the tiny seed sample, not a
real effect. The ranking becomes worth interpreting only on the full
365-day backfill (workstream 5's walk-forward CV will show whether any
context feature carries stable weight).

## FinBERT text representation (0.3.0 workstream 2)

ProsusAI/finbert is used as a *frozen* feature extractor: one 768-dim
vector per article (CLS pooling by default, attention-masked mean pooling
on request; eval mode under `torch.no_grad()`, no fine-tuning).
`signal_lab.models.embeddings` compares it against the 0.2.0
representation in a strict ablation: same articles, same temporal
train/test split, same classifier family (class-weighted logistic
regression, threshold 0.5) in every arm, so differences measure the
representation, not the protocol.

**Ablation (694 labeled rows, 485 train / 209 test, positive rate 10.2%,
windowed 0.3.0 labels):**

| Arm | PR-AUC | F1 | Precision | Recall |
|-----|--------|----|-----------|--------|
| TF-IDF + LM lexicon | 0.1395 | 0.0816 | 0.1176 | 0.0625 |
| FinBERT-only | 0.1643 | 0.0444 | 0.0769 | 0.0312 |
| Combined (TF-IDF + lexicon + FinBERT) | 0.1564 | 0.0000 | 0.0000 | 0.0000 |

Chance level is ~0.10 (the positive rate). The FinBERT-only arm gains a
modest +0.025 PR-AUC over TF-IDF+lexicon but scores a lower F1 at the
fixed 0.5 threshold; the combined arm's F1 collapses to 0.000 because
with only 485 training rows and ~1,800 features the classifier predicts
no positives at threshold 0.5. All three arms sit near chance, consistent
with the 0.2.0 finding that data scale, not features, is the binding
constraint: a 768-dim frozen extractor cannot rescue 694 noisy labels,
and the honest verdict is that FinBERT does not buy a meaningful signal
on this corpus.

**Optional-dependency design.** torch + transformers are NOT hard
requirements: they are absent from `requirements.txt` and never imported
at module scope (only lazily inside the extractor). Behavior matrix:

- Deps missing: stage 3 logs `[m3] FinBERT unavailable (torch not
  installed); finbert/combined arms skipped, falling back to
  TF-IDF+lexicon`, and the `finbert_ablation` JSON block reports
  `"available": false` with the reason. The TF-IDF arm still reports, so
  the block is never empty.
- Deps present, weights never downloaded: the opt-in `--finbert` run
  downloads ~440MB of `ProsusAI/finbert` weights into the standard
  Hugging Face cache (one loud log line first); a failed download
  (offline, hub outage) falls back the same clean way.
- The seeded demo and `signallab doctor` work fully offline. `doctor`
  gained a `finbert embeddings (optional)` check reporting installed /
  weights-cached / unavailable; the check passes whenever the probe runs,
  since missing FinBERT is a normal, supported state.

**Caching.** Per-article embeddings are cached as `.npy` files under
`~/.cache/signal_lab/finbert` (override with `SIGNAL_LAB_FINBERT_CACHE`),
keyed by SHA-256 of the whitespace-normalized text, or by an explicit key
such as the article URL; re-runs never recompute. The ablation itself is
opt-in (`--finbert` flag or `SIGNAL_LAB_FINBERT=1`), so nothing downloads
unannounced.

**Stage-3 JSON contract.** Every run emits a `finbert_ablation` block
(`available`, `model`, `embedding_dim`, `pooling`, `n_train`, `n_test`,
`positive_rate`, `arms`, plus `reason`/`cache_dir` when applicable). The
`label_config` block from workstream 1 is untouched.

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
  moves with random volatility. 0.3.0 widens the window and filters
  quiet days, but the labels are still market-derived, not human.
- Market-context features are present (0.3.0 workstream 3), but the
  sector-relative-strength leg currently falls back to SPY because sector
  ETF prices are not in the backfill; rows with no benchmark leg are
  neutralized to 0.0 and counted (878 rows in the 0.3.0 backfill) rather
  than silently dropped. See the Features section and
  [docs/research-note-0.3.0.md](research-note-0.3.0.md).
- Probability calibration exists (0.3.0 workstream 4): Platt scaling
  improves ECE (0.110 to 0.077 on the holdout); isotonic regression hurts
  (0.135) on 121 calibration points and is not used. Confidence abstention
  was tested and does not help: precision is flat at 0.067 across tau in
  [0.10, 0.50], and the backtest gate collapses event coverage without
  improving Sharpe. Full negative results in
  [docs/research-note-0.3.0.md](research-note-0.3.0.md).
- Evaluation is 5-fold purged walk-forward CV with embargo (0.3.0
  workstream 5): mean PR-AUC 0.170, 95% CI [0.019, 0.321]; mean F1 0.018,
  95% CI [-0.039, 0.075]. Fold 1 was unscorable (no positives in the early
  training window). The single-split number hid major regime instability.
- Gold-set human labels are still pending, so the weak-label agreement
  rate is unmeasured.
- See [docs/limitations.md](limitations.md) and the 0.3.0 roadmap
  ([docs/roadmap-0.3.0.md](roadmap-0.3.0.md)).

## Citation

Finance word lists derived from the Loughran-McDonald Master Dictionary:
Loughran, T. and McDonald, B. (2011). "When Is a Liability Not a Liability?
Textual Analysis, Dictionaries, and 10-Ks." *Journal of Finance* 66(1),
35-65. Used under the SRAF academic-research terms; commercial use requires
a license from the authors.
