# Program status: research program CLOSED (2026-10-01)

The leakage-safe research program (Phases 0-3 plus the gate-2 retest
attempt) is closed. Gate-1's NO-GO is the final verdict; no further
gate runs are planned.

## Final verdicts

- **Gate 1 (completed 2026-10-01): NO-GO.** Text over price-only:
  mean paired PR-AUC difference +0.0035, 95% CI [-0.0091, 0.0160],
  paired t p = 0.48, on 693 dev rows. Full report: docs/GATE.md.
- **Gate 2 (attempted 2026-10-01): no verdict.** The pre-registered
  retest (FinBERT primary text arm, widened to 100+ tickers) was
  blocked first by GDELT DOC API hard-throttling (HTTP 429, ~100%
  failure rate, ~100h projected for the full widening) and then by
  bandwidth limits on the bulk GKG path (~300+ GB at 1.4-2.2 MB/s).
  The pre-specified fallback was executed: docs/NEGATIVE_RESULT.md.
  Pre-registered gate-2 code (FinBERT arm, widening scripts,
  leakage-checked benchmark) remains on main for any future run.

## Integrity checks completed during gate-2

- SPY-gap check (EXP-029): the SPY backfill changed 0 cells across
  all 49 columns of the original gate-1 frame. Gate-1 results stand
  exactly as reported.
- Leakage check (EXP-028): the early-fold price signal was noise,
  not harness leakage (fold CIs contain chance; collapsed folds had
  6 positive samples each).

## Frozen periods

The test period (2026-07-01 through 2026-08-31) and the confirmation
period (t0 >= 2026-09-01) were never evaluated, read, or scored at
any point in the program. They remain frozen.

## What comes next

The program's engineering assets (purged walk-forward harness,
versioned targets, point-in-time price/volume features, event-study
machinery, cited explainer) continue as product work for the
explanatory workbench. Improvements extracted from the open-source
survey are logged in docs/EXPERIMENT_REGISTRY.md as APPLIED items,
not gate experiments.
