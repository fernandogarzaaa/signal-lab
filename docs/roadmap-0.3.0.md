# Signal Lab 0.3.0 Scope: Robust Model

Status: draft (2026-09-30)
Depends on: 0.2.0 published (full 365-day backfill, all six stages re-run, re-seeded)

## Objective

Make the stage-3 sentiment classifier genuinely more robust: cleaner training
signal, richer representation of financial text, market-aware features,
calibrated outputs, and a validation harness trustworthy enough to believe.
Target: a model whose evaluation we can defend in an interview, not just a
higher number.

## Baseline (0.2.0, for comparison)

- Logistic regression on TF-IDF + 8 Loughran-McDonald lexicon features via a
  shared `featurize()` used by train, predict, and artifact scoring.
- Weak labels: 1 if the ticker's next-trading-day abnormal return vs SPY lands
  in the top decile, else 0. ~10% positive rate.
- 337 labeled rows (n_train=235, n_test=102), single time-based split.
- Candidates compared: plain, class_weights, resampling, threshold_tuning,
  lightgbm_balanced. Best single-split PR-AUC ~0.16 (barely above chance).
- JSON output contract: baseline / challenger / imbalance_comparison /
  majority_baseline / n_train / n_test.

## Workstreams

### 1. Label quality (do first — it changes everything downstream)

The weak labels are the noisiest part of the system. A single next-day return
mixes real news-driven moves with random jumps.

- Multi-day labeling window: label on cumulative abnormal return over t+1..t+3
  instead of t+1 only. News takes time to digest; single-day labels punish the
  model for slow reactions.
- Attention filter: only label articles for ticker-days with sufficient news
  presence (e.g., >= 2 distinct articles or top-quartile daily article count
  for that ticker). Quiet-day labels are mostly noise.
- Gold set: hand-label 100-200 articles (does this read positive/negative for
  the company?) and report weak-label agreement rate. This quantifies label
  noise instead of guessing at it.
- Keep the 0.2.0 labeling as a documented baseline so the improvement is
  measurable, not just claimed.

Acceptance: new labeling code in `models/`, agreement rate reported in the
model card, positive rate stable (5-15%), stage-3 JSON gains a `label_config`
block describing the labeling used.

### 2. FinBERT text representation (the headline upgrade)

- Add ProsusAI/finbert embeddings as features alongside TF-IDF (ablation:
  TF-IDF-only vs FinBERT-only vs combined, same data, same splits).
- Lazy/optional dependency: torch + transformers + a ~440MB model download is
  heavy for the npm bundle. Design: FinBERT features computed when the model
  is available, clean fallback to TF-IDF+lexicon when not. The seeded demo
  must still work offline.
- Cache embeddings per article URL so re-runs don't recompute.

Acceptance: ablation numbers in the model card, offline fallback tested,
`doctor` reports FinBERT availability as a check.

### 3. Market-context features

The model currently reads only the article. Add features known at prediction
time (no leakage):

- Volatility regime: trailing-20d realized volatility percentile for the ticker.
- Momentum: trailing-5d and trailing-20d returns.
- Sector-relative strength: ticker return minus sector ETF return, trailing 5d.
- Article metadata: source-domain tier, word count, hour-of-day bucket.

Acceptance: feature-importance table in stage-3 output, leakage audit noted in
the model card (every feature timestamped <= article publish time).

### 4. Calibration and abstention

Robustness means knowing when not to bet.

- Calibrate predicted probabilities (Platt scaling or isotonic) on a held-out
  validation fold; report expected calibration error (ECE) and a calibration
  curve in the model card.
- Abstention: the backtest only trades predictions above a confidence
  threshold; report coverage (% of articles traded) alongside returns.
- Acceptance: ECE reported before/after calibration; backtest JSON includes
  coverage and calibrated-vs-uncalibrated comparison.

### 5. Validation discipline

- Replace the single time-based split with purged walk-forward CV: 5 folds,
  embargo gap (e.g., 5 trading days) between train and test so no information
  leaks across the boundary.
- Report per-fold row counts and positive counts, mean/variance/CIs for
  PR-AUC and F1. Never select a "best" model from one fold.
- Acceptance: stage-3 JSON gains a `cv` block; model card shows the full
  per-fold table.

### 6. Model card and docs

- Update the model card: intended use, training data, labeling method,
  evaluation results with CIs, known limitations (label noise, regime
  dependence, small-sample caveats).
- Update the interview cheat sheet with the 0.3.0 story.
- Refresh the README architecture section if the pipeline changes.

## Sequencing

1. Workstream 1 (labels) first, on the 0.2.0 dataset.
2. Workstreams 2+3 (FinBERT, context features) in parallel once labels are set.
3. Workstream 5 (validation) built alongside 2+3, used to evaluate them.
4. Workstream 4 (calibration) after the final feature set is chosen.
5. Workstream 6 (docs) throughout, finalized at the end.

## Non-goals for 0.3.0

- Training a neural classifier from scratch (FinBERT is used as a frozen
  feature extractor).
- Real-time/streaming inference.
- Additional asset classes (options, crypto, forex).
- PySpark migration (DuckDB stays).

## Risks

- FinBERT weight vs npm bundle size: mitigated by lazy optional dependency.
- New labels invalidate 0.2.0 comparisons: expected; report both label
  configs side by side during the transition.
- Data hunger: FinBERT + more features on ~337 rows risks overfitting.
  Mitigated by workstream 5 (honest CIs will show it) and continued backfill.
- GDELT throttling on further backfill: same retry machinery as 0.2.0.

## Estimates (rough, in focused work sessions)

- WS1 labels: 2-3 sessions (gold-set labeling is the long pole; can be
  parallelized as a background task).
- WS2 FinBERT: 2 sessions (packaging/fallback is the tricky part).
- WS3 context features: 1-2 sessions.
- WS4 calibration: 1 session.
- WS5 validation: 2 sessions.
- WS6 docs: 1 session.
- Release (PR, CI, publish): 1 session.

Total: roughly 10-13 sessions after 0.2.0 ships.

## Definition of done

- All six workstreams complete with acceptance criteria met.
- `pytest` green, CI green, PR merged (one PR, squash, never direct to main).
- All six stages re-run on final data; seed snapshot regenerated.
- Fresh `npm pack` + fresh install verified through real entrypoints
  (setup, doctor, seed, all stages, dashboard).
- `signallab@0.3.0` published only after the above; registry version and
  tarball digest confirmed.
