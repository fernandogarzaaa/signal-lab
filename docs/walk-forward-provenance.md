# Walk-forward provenance (0.5.1)

Every PR-AUC number below is listed with what produced it. Numbers without a
provenance block are not comparable across runs; the walk-forward writer has
embedded a `provenance` block since 0.5.1, so new runs are self-describing.

## Numbers currently in circulation

| Number | Mean PR-AUC | Folds | Classifier | Representation | Labels | Provenance |
|---|---|---|---|---|---|---|
| Research note 0.5.0 headline | 0.1409 | 5: 0.1435 / 0.0949 / 0.1625 / 0.0775 / 0.2262 | LogReg balanced | finbert_ctx | windowed (weak) | Partial: purge=3d, embargo=3d. Exact data snapshot, cache state, and code revision were not recorded. |
| Independent re-run of the 0.5.0 protocol | ~0.1417 | 5: 0.1451 / 0.0948 / 0.1628 / 0.0771 / 0.2289 | LogReg balanced | finbert_ctx | windowed (weak) | Partial: same protocol, different data state. Within noise of the headline; the fold-level deltas (e.g. 0.1435 vs 0.1451) are not explained. Do not treat either as exact. |
| HGB experiment (0.5.0 negative result) | 0.1812 | 5 | HistGradientBoosting | finbert_ctx | windowed (weak) | Documented in the 0.5.0 note as overfit: **do not use**. Reverted to LogReg. Present in `data/artifacts/walk_forward.json` on the dev machine before 0.5.1 (git-ignored local state, since overwritten). |
| npm seed `walk_forward.json` | 0.1699 | 4 (folds 2-5; fold 1 absent, no skip record) | unknown | unknown | unknown | Committed 2026-10-01 (0.3.0 WS6). No representation, classifier, or label block. **Demo seed only; do not cite as a result.** |
| `walk_forward_finbert.json` arm (f) | 0.1444 | 3 | LogReg | finbert+ctx | windowed (weak) | Local experiment artifact (git-ignored). Different fold count; not the official protocol. |
| 0.5.1 fresh run (current data) | 0.127 | 4 scored, 1 skipped (fold 1 single-class) | logreg_balanced | finbert_ctx | windowed (weak) | Full provenance block in `data/artifacts/walk_forward.json`. Embedding cache: 632/632 unique texts cached, 0 missing. Mean ROC-AUC 0.4841 (ranking no better than chance). |
| 0.5.1 fresh run, TF-IDF arm | 0.1016 | 4 scored, 1 skipped | logreg_balanced | tfidf | windowed (weak) | Full provenance block. Mean ROC-AUC 0.461. |

## Discrepancies we are not papering over

1. The 0.5.0 headline folds (0.1435 / ...) and the re-run folds (0.1451 / ...)
   differ in the third decimal. Both used the same protocol on paper; the
   exact dataset snapshot and code revision behind each were not recorded, so
   the delta is unexplained. The headline number stands as published with this
   caveat, not as a reproducible exact value.
2. The npm seed's 0.1699 has no provenance at all. It renders in the dashboard
   diagnostics view; treat it as a rendering placeholder, not evidence.
3. The HGB 0.1812 was explicitly rejected in the 0.5.0 note ("Do not use") yet
   sat in the local `walk_forward.json`. It has been overwritten by the 0.5.1
   provenanced run.

## Label scheme: windowed remains the default

`windowed` (cumulative abnormal return vs SPY, 3-day window, 0.90 quantile,
attention filter) has been the runtime default since 0.3.0 WS1 and remains
the default in `LabelConfig` and every `--label-scheme` CLI flag.

A 0.4.0 planning note (`docs/overnight-0.4.0-plan.md`) recorded a 3-fold
experiment favoring `baseline` (0.1412 vs 0.0882) with the line "sticking with
baseline scheme". That decision was never implemented: no code change, no
config change, no follow-up run. The default stayed `windowed`, and the model
card documents `windowed` as the current scheme. This document records the
status quo explicitly so the plan note is not mistaken for a decision.

## Embedding cache state

- Location: `~/.cache/signal_lab/finbert` (override: `$SIGNAL_LAB_FINBERT_CACHE`).
- Machine-local, git-ignored, not shipped with the npm package.
- Dev machine: 815 `.npy` files; the 0.5.1 finbert_ctx walk-forward saw
  632/632 unique texts cached (0 missing).
- `scripts/fill_finbert_cache.py` fills the cache idempotently (needs torch +
  transformers + ~440MB download on first use). The fill was run to completion
  for the labeled set on the dev machine; on any other machine the cache
  starts empty and `finbert_ctx_matrix` raises `MissingEmbeddingError`
  instead of silently zero-filling.
- The 0.5.0 note's "100% coverage on the 902 labeled rows (813 unique texts)"
  describes the dev machine's cache at that time. It is a machine-state claim,
  not a property of the package.

## Feature-space note (monitor)

The shipped `model.pkl` is `lightgbm_balanced` trained on TF-IDF (5000) + 8
lexicon features = 5008 dims, without the 7 market-context features. Its exact
training run predates the context-feature pipeline and its provenance was not
recorded. The monitor now adapts: it scores on exactly the feature space the
loaded model expects (verified against `model.n_features_in_`) and raises
loudly on anything else.
