# Frozen Evaluation Periods

The dataset is split into three frozen calendar periods. The boundaries are
constants in `src/signal_lab/validation/periods.py` and must not be moved
without a full re-validation (they define what "out-of-sample" means).

| Period       | Range                    | Purpose                                              |
|--------------|--------------------------|------------------------------------------------------|
| Development  | t0 <= 2026-06-30         | All model selection, tuning, walk-forward, training. |
| Test         | 2026-07-01 .. 2026-08-31 | Untouched until the Phase 3 gate. One evaluation.    |
| Confirmation | t0 >= 2026-09-01         | Frozen holdout. Never enters any development path.   |

## Rules

1. **Selection never touches test.** Model selection (Phase 1: `select_model`)
   runs on a single purged validation split *inside* development. The test
   period is not looked at, not even for "just checking", until the Phase 3
   gate configuration is finalized.

2. **Confirmation never enters development.** `periods.check_no_confirmation`
   raises `ConfirmationLeakError` if any row with t0 >= 2026-09-01 reaches
   `make_splits`. `run_walk_forward` filters to the effective dev range
   before splitting, so the guard only fires on genuine misuse.

3. **The test period is evaluated once.** `final_train_and_test` requires
   `confirm_test_eval=True`, an explicit acknowledgment that this touches
   the untouched test period. It is never used for selection, tuning,
   thresholds, or calibration.

4. **Purge is against the test range.** In `final_train_and_test`, dev rows
   whose label window [t0, t1] reaches into the test range are purged from
   final training. A dev row whose label embeds test-period returns must not
   train the final model.

## Why frozen

Moving the boundaries after seeing results is a form of selection bias:
the "out-of-sample" claim only holds if the sample was truly untouched when
decisions were made. The constants are frozen; the Phase 3 gate is the
single authorized test-period evaluation.

## Purge scope note (deliberate deviation)

The Phase 1 spec text said to purge training rows against "any test range
j <= k". The implementation purges only against the *current* fold's test
range (j == k): a train row is dropped iff its t1 >= test_start_k. Purging
against earlier folds' ranges would drop every train row with t0 in blocks
1..k-1, collapsing the expanding window to block 0 (verified empirically:
fold train sets went 29/29/29 instead of 29/61/92). Earlier blocks' labels
use only returns strictly before test_start_k, so they carry no test-period
information and are legitimate point-in-time training data. This is
documented in the `make_splits` docstring and flagged for maintainer review
in the Phase 1 PR.
