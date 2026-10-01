# Signal Lab: Leakage and Validity Audit (Phase 0)

Date: 2026-10-01. Auditor: lead ML/research engineer (Cookie).
Scope: `src/signal_lab/` as of commit `2fdfe37`, plus `data/signal_lab.duckdb`
as shipped. Nothing in this document assumes docs or tests prove correctness;
every claim below was checked against the code or re-run.

## 1. Data flow

```
GDELT 2.1 DOC API ──> news_raw (DuckDB) ──┐
yfinance daily OHLCV ──> prices_daily ─────┤
                                           ▼
                    build_and_train.build_dataset()
                      ├─ entity extraction (rules, extract_baseline)
                      ├─ context features (7, pre-publication only)
                      ├─ weak labels (labels.build_labels)
                      └─> labeled frame (text, ticker, published_at, label)
                                           ▼
                    ┌─ models.train(): single 70/30 temporal split
                    │   → vectorizer.pkl, model.pkl (BEST test PR-AUC),
                    │     scored_news.csv (in-sample scores, ALL rows)
                    │
                    └─ models.walk_forward.run_walk_forward(): 5 purged +
                        embargoed folds, retrains per fold, TF-IDF fit per
                        fold. This is the honest evaluation path.
                                           ▼
                    backtest.run_backtest(): reads scored_news.csv,
                    walk-forward p95 threshold per fold, strategy vs
                    buy-and-hold vs random.
```

## 2. Dataset (verified 2026-10-01)

| Item | Value |
|---|---|
| `news_raw` rows | 1,709 |
| `prices_daily` rows | 5,391 (11 tickers incl. SPY) |
| Price range | 2024-10-15 to 2026-09-30 |
| News range | 2025-10-15 to 2026-09-30 |
| Universe | 10 large-cap US equities, hand-picked (AAPL MSFT NVDA GOOGL META AMZN TSLA JPM JNJ XOM) + SPY benchmark |
| Labeled rows (windowed, default) | ~902 (varies with attention filter), positive rate ~10% |

Limits: single regime (2025-2026 US large caps), GDELT 429 throttling left
25 of 69 retry windows empty, survivorship/selection bias by construction.

## 3. Label and feature definitions

**Labels** (`src/signal_lab/models/labels.py`, `LabelConfig`):
- `windowed` (default): label 1 iff the ticker's cumulative abnormal return
  vs the benchmark over trading days t+1..t+`window_days` (default 3) lands
  in the top decile (`quantile=0.90`); attention filter keeps only
  ticker-days with >= 2 articles or busier than 75% of the ticker's days.
- `baseline`: next-trading-day top-decile abnormal return, no attention
  filter. The 0.2.0 scheme, kept for comparison.

**Features**:
- Text: TF-IDF (5000, 1-2 grams, English stopwords, sublinear) or frozen
  FinBERT CLS (768-dim, cache-only, `MissingEmbeddingError` on misses).
- Lexicon: 8 Loughran-McDonald features.
- Context (7): `ctx_vol20_pct`, `ctx_mom5`, `ctx_mom20`, `ctx_sector_rel5`,
  `ctx_src_tier`, `ctx_word_count`, `ctx_hour_bucket`. All price inputs use
  the latest trading day *strictly before* the publish date
  (`searchsorted(..., side="left") - 1`); a synthetic-spike test pins this.

## 4. Validation: explicit checks

### 4a. Is the model trained once on all data then "walked forward"? NO.

`run_walk_forward` (`src/signal_lab/models/walk_forward.py`) retrains the
logistic regression from scratch in every fold and fits a fresh TF-IDF
vectorizer on train texts only. The dense context features are computed
per-article from pre-publication data (no fitting). The FinBERT cache is a
frozen per-text extractor: cache hits do not depend on labels or folds, so
populating the cache on the full text set is not label leakage (same
status as a fixed tokenizer). Verified: `vec.fit(texts.iloc[tri_eff])`
inside the fold loop (walk_forward.py), `clf.fit(Xtr, ytr)` per fold.

### 4b. Is any model or threshold chosen on the test set? YES, in the production path.

- **Model selection on test PR-AUC.** `run_pipeline`
  (`src/signal_lab/models/build_and_train.py:246`):
  `best_name = max(scored)[1]` picks the highest *test* PR-AUC among the
  fitted candidates and persists it as `model.pkl`. The persisted model's
  identity is a test-set decision. The walk-forward evaluation is not
  affected (it always uses `logreg_balanced`), but the shipped artifact is.
- **Threshold tuning is clean.** `tune_threshold` is fit on train scores
  only (`models/__init__.py`, `threshold_tuned_on: train`).

### 4c. Label-construction leakage (confirmed, mild)

`build_labels` (`src/signal_lab/models/labels.py:250`) computes the
top-decile cutoff on the **full** labeled sample *before* any split:
`cutoff = df["abn_ret"].quantile(0.90)`. Train labels therefore depend on
test-period returns. Measured on the current data: full-sample cutoff
0.03486 vs first-half-only cutoff 0.03334 (delta 0.00151). Articles near the
boundary change labels depending on future data. Mild, but it is leakage,
and the fix (per-fold cutoff from train data only) is a Phase 1 item.

### 4d. Attention-filter leakage (confirmed, mild)

`apply_attention` (`labels.py:235`) computes per-ticker day-count quantiles
over the **full** sample before splitting. Whether a train article survives
into the dataset depends on test-period article counts. Mild; fix in
Phase 1 (compute the filter on train data, apply the rule to all rows).

### 4e. Single-split train() has no purge/embargo

`models.train()` (`src/signal_lab/models/__init__.py:228`) uses a plain
70/30 `temporal_split`. Train articles adjacent to the boundary have label
windows (t+1..t+3) reaching into the test period. The walk-forward path
purges these; the production-training path does not. Any metric reported
from `train()` is boundary-optimistic.

### 4f. In-sample scores feed the backtest and explainers (confirmed)

`run_pipeline` (`build_and_train.py:255`) scores **every** article with the
fitted model, including the 70% the model trained on, and writes
`scored_news.csv`. `run_backtest` (`src/signal_lab/backtest/run_backtest.py:66`)
reads that file. Consequences:
- The per-fold p95 "sentiment spike" thresholds are computed from
  probabilities that are in-sample for train-period articles.
- Event detection, the explain UI (`explain_move.py`), the analogues view,
  and the RAG headline lookup all consume in-sample probabilities.
- The backtest's walk-forward structure is about the *threshold*, not the
  *model*: the model was trained on data spanning the backtest's evaluation
  periods. Reported backtest numbers are optimistic by construction.

The backtest engine itself (`src/signal_lab/backtest/__init__.py`) is
mechanically sound: `merge_asof` on `asof <= date` means a signal stamped
`asof=d` cannot capture the return from d-1 to d, and a behavioral test
pins this. But "honest asof stamping is the caller's responsibility," and
the caller feeds it in-sample scores.

### 4g. What is NOT leakage (checked)

- Context features: pre-publication only, structurally (as-of day is
  strictly before publish date). Pinned by a synthetic-spike test.
- FinBERT embeddings: frozen, per-text, label-independent.
- TF-IDF in walk-forward: fit per fold on train texts.
- Event-window news for the explainer: by construction from the event
  date (lookahead bias would be using later news; the UI now labels this).

### 4h. Correlated samples (documented, not fixed)

Same-ticker same-day articles share a label (`models/__init__.py`
docstring). Folds treat them as independent; standard errors are
understated. Phase 1 dedupes or down-weights.

## 5. Test-set contamination risks, ranked

1. Backtest consumes in-sample model scores (4f). HIGH impact on
   strategy numbers; the event detector sees train-fit probabilities.
2. Model selection on test PR-AUC for the shipped artifact (4b). MEDIUM;
   affects which model ships, not the walk-forward numbers.
3. Global label cutoff (4c) and global attention filter (4d). LOW-MEDIUM;
   measured deltas are small but nonzero.
4. No purge in `train()` boundary (4e). LOW; affects only single-split
   metrics, not walk-forward.

## 6. Migration risks (Phase 1 rebuild)

- The production path (`train()` + `model.pkl` + `scored_news.csv` +
  monitor/explainer) and the honest path (`run_walk_forward`) currently
  disagree on methodology. Phase 1 must converge them: one splitter, one
  label builder, one scoring contract.
- `scored_news.csv` is load-bearing for the dashboard, monitor, analogues,
  and RAG. Replacing in-sample scores with OOS (fold-test) scores changes
  every downstream number; the UI must not silently mix old and new.
- The monitor's `model.pkl` is `lightgbm_balanced` on 5008 TF-IDF+lexicon
  features with no recorded provenance (see walk-forward-provenance.md).
  Phase 1 re-trains it under the new protocol or documents why not.
- FinBERT cache is machine-local (`~/.cache/signal_lab/finbert`); CI and
  fresh machines cannot reproduce finbert_ctx runs. The TF-IDF arm must
  stay runnable everywhere.

## 7. Baseline numbers (the honest starting point, 0.5.1)

finbert_ctx walk-forward: mean PR-AUC **0.127** (4 folds, 1 skipped),
mean ROC-AUC **0.4841** (ranking no better than chance).
TF-IDF arm: PR-AUC **0.1016**, ROC-AUC **0.461**.
Backtest (0.2.0): sentiment +1.1% vs buy-and-hold +14.5%, paired p = 0.31,
not significant. The backtest number is additionally optimistic per 4f.

## 8. Phase 1 work items (from this audit)

1. Per-fold label cutoff and attention filter (fix 4c, 4d); regression
   tests asserting train-only label construction.
2. Purge/embargo in all training paths, or delete the unpurged path (4e).
3. Model selection on validation only; never persist a test-chosen model (4b).
4. OOS-only scoring artifact: replace in-sample `scored_news.csv` with
   fold-test (or refit-on-past) probabilities everywhere downstream (4f).
5. Dedupe/down-weight same-ticker same-day articles (4h).
6. Leakage tripwires: shuffled-labels-near-chance test, injected
   future-feature test, per-fold overlap assertions, scaler/calibrator
   fit-window tests.

### Phase 1 completion notes (2026-10-01)

All six items shipped in `src/signal_lab/validation/` (timing, periods,
splits, labels, dedupe, point_in_time) with rewired
`models/{labels,walk_forward,build_and_train}.py`. 209 tests pass.

Findings worth recording:

- **4c/4d measured delta: 0.00151** on 2026-10-01 data (global vs per-fold
  cutoff). Small on this dataset, but the mechanism is fixed regardless.
- **Deliberate purge-scope deviation** (see docs/PERIODS.md): purge is
  j == k only, not j <= k as the spec text said. Purging earlier folds
  collapses the expanding window (29/29/29 vs 29/61/92 rows). Flagged for
  maintainer review in the Phase 1 PR.
- **attention_enabled bug**: `run_walk_forward` and `final_train_and_test`
  applied the attention filter even when disabled. Fixed; both use an
  all-ones mask when `attention_enabled=False`.
- **Cut snapping**: positional cuts in `make_splits` now snap to t0
  boundaries so a day's rows are never split across train/test.
- **Leakage canary**: injected future text in the purge zone scores
  PR-AUC 1.0 without purge, 0.5 (chance) with purge
  (tests/test_leakage.py). The `purge=False` escape hatch exists only for
  this test.
- **Real-data smoke**: 3-fold purged walk-forward on 1,227 rows
  (816 dev) gives mean PR-AUC 0.165, consistent with the 0.127 baseline
  direction. One fold skipped (single-class test set), handled gracefully.
- **ctx_asof added** to `build_context_features` output: the point-in-time
  anchor (latest trading day strictly before publication) now flows through
  `build_dataset` and is asserted by `point_in_time.check_frame`.
