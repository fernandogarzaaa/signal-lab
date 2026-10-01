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

**Tested-experiment count: 28** (26 baseline + 2 gate-2).
