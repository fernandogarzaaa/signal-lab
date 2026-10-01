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
