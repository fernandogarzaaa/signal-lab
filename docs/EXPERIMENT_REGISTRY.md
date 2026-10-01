# Experiment Registry

Every tested model, feature set, horizon, and threshold is logged here.
Results are labeled EXPLORATORY or CONFIRMATORY. The registry was created
for the gate-2 retest (Phase 5 of the original program was never built);
pre-registry experiments are reconstructed from merged PRs and committed
artifacts and marked as a lower-bound estimate.

## Pre-registry baseline (lower bound, reconstructed 2026-10-01)

| IDs | Experiment | Status |
|---|---|---|
| EXP-001..015 | Gate 1: 5 models x 3 variants (A/B/C), dev walk-forward | EXPLORATORY |
| EXP-016 | Gate 1: calibration diagnostic (Platt/isotonic) | EXPLORATORY |
| EXP-017 | Gate 1: abstention diagnostic | EXPLORATORY |
| EXP-018..020 | Walk-forward label schemes: weak / jev / jev-conf06 | EXPLORATORY |
| EXP-021..023 | FinBERT model experiments: logreg / HGB / logreg+PCA50 | EXPLORATORY |
| EXP-024..026 | FinBERT ablations (0.3.0/0.4.0): TF-IDF / FinBERT / combined | EXPLORATORY |

**Tested-experiment count at registry creation: 26.** Ad-hoc runs before
the research program that left no committed trace are not counted.

## Gate-2 run (all EXPLORATORY unless noted)

| ID | Date (UTC) | Experiment | Result pointer |
|---|---|---|---|
| EXP-027 | 2026-10-01 | Gate-2 retest: FinBERT primary arm, TF-IDF arm, widened dev data; pre-registered in docs/GATE_2_PREREGISTRATION.md | docs/GATE_2.md (pending) |
| EXP-028 | 2026-10-01 | STEP 2: leakage check on early-fold price signal (purge/embargo re-run, shuffled labels, injected future feature) | report section (pending) |
| EXP-029 | 2026-10-01 | STEP 0a: SPY-gap integrity check. Rebuilt the original gate-1 frame (909 pre-widening rows) with SPY truncated to >= 2024-10-15 (pre-backfill state) vs full backfilled SPY. Result: 0 differing cells across all 49 columns; the fallback market leg was used on 325 pre-2024-10-15 dates but none fed any label or feature. Gate-1 results unchanged. Script: scripts/gate2_step0_spy_check.py, scripts/gate2_step0_spy_fulldiff.py | this row |
| EXP-030 | 2026-10-01 | STEP 1: bulk news-source access verification (EXPLORATORY). GDELT bulk GKG verified: free, no auth, no documented rate limit, HTTPS, predictable 15-min slots, 27-col TSV, Extras carries PAGE_PRECISEPUBTIMESTAMP + PAGE_TITLE. Measured throughput 1.4-2.2 MB/s from this box -> pre-registered 3-year pull projects to 38-78h, infeasible in-session. BigQuery gdelt-bq.gdeltv2.gkg exists but needs GCP creds (unavailable). Hard stop triggered -> docs/NEGATIVE_RESULT.md fallback. | docs/NEGATIVE_RESULT.md |

**Tested-experiment count: 30** (26 baseline + 4 gate-2).

## Extraction program (APPLIED product improvements, 2026-10-01)

The research program is CLOSED (see docs/PROGRAM_STATUS.md). The
items below are product improvements for the explanatory workbench,
extracted from the open-source survey. They are not gate experiments
and carry no predictive claims.

| ID | Date (UTC) | Item | Result pointer |
|---|---|---|---|
| APP-000 | 2026-10-01 | Program closure note | docs/PROGRAM_STATUS.md |
| APP-001 | 2026-10-01 | Split-conformal abstention (src/signal_lab/models/conformal.py): rolling-window calibration per fold, abstain on empty/non-singleton sets; replaces failed ad-hoc margins | tests/test_conformal.py (10 tests) |
| APP-002 | 2026-10-01 | CAR significance tests (src/signal_lab/stats/car_tests.py): BMP / adj-BMP (Kolari-Pynnonen) / GRANK wired into event_study; near-zero-variance estimation windows dropped, not zero-filled | tests/test_car_tests.py (12 tests) |
| APP-003 | 2026-10-02 | TabPFN zero-tuning benchmark arm (src/signal_lab/models/tabpfn_arm.py, tabpfn==2.0.6 last fully-local release): purged/embargoed walk-forward reusing the validation harness; per-fold PCA-50 fit on train only, dedupe-first, point-in-time conditioning asserted per fold. Dev result (4 folds, fold 1 skipped at 43 rows): PR-AUC 0.1184 +/- 0.0704 vs no-skill 0.1481, ROC-AUC 0.3517, F1 0.0; below tuned logreg (0.2079) and LightGBM (0.1925). Honest negative: zero tuning does not beat noise on this task; the arm stands as the zero-tuning floor for future models | tests/test_tabpfn_arm.py (6 tests); data/artifacts/tabpfn_arm.json |
| APP-004 | 2026-10-02 | Fail-loud baseline/event-window leakage assertion (src/signal_lab/stats/leakage_guard.py; pattern from jiamingpan/agent-quant-research, reimplemented): BaselineLeakageError (AssertionError subclass, -O-safe) raised when any baseline label is on/after the event-window start. Wired into event_ar_panel (per-event panel check), _scar_frame (re-checks any hand-built panel before the estimation/event split), and explain_move._price_block (trailing 60-day abnormal-z baseline must end before the event day) | tests/test_leakage_guard.py (12 tests: guard unit incl. boundary, hostile panel through cross_sectional_tests, clean-path integration, explainer call shapes) |
| APP-005 | 2026-10-02 | gdelt-client migration (src/signal_lab/ingest/gdelt_client.py, gdelt-client==0.2.2): hand-rolled GDELT DOC requests loop replaced by Filters/article_search; strict query-template parsing (unknown templates fail loudly), same day bounds (start-of-day..end-of-day) and max_records semantics, tenacity retries on 429/5xx with the old backoff posture (15s base, 120s cap). Code health only, not a throttle fix; checkpointing (fetch_log), politeness sleeps, and explicit gap logging unchanged. Documented deviation: no sort=datedesc (client exposes no sort param; ordering unused downstream) | tests/test_gdelt_client.py (10 tests: template-to-Filters mapping, loud rejection of unknown queries, row mapping, empty/URL-less rows, snippet fn, RateLimitError propagation, backoff config, fetch_log error bookkeeping) |
