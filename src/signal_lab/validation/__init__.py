"""Validation integrity framework (Phase 1).

The single home for everything about *when* data may be used:

- ``timing``: per-sample t0 (first tradable time after publication) and t1
  (t0 + H trading days), trading-calendar aware.
- ``periods``: the frozen research periods (development / test /
  confirmation) with exact timestamps, plus a guard that refuses to let
  confirmation-period data into any development path.
- ``splits``: purged + embargoed walk-forward splitter (expanding and
  rolling windows, configurable retrain frequency, deterministic).
- ``labels``: per-fold label construction. The label *cutoff* and the
  attention-filter quantiles are computed on train data only; the old
  global-cutoff behavior is kept in ``models.labels`` for CLI compatibility
  but is NOT used by any validation path.
- ``dedupe``: same-ticker same-t0 dedupe / down-weighting.
- ``point_in_time``: per-feature observation/publication/availability
  declarations.

Rule of the module: no function here may look at data dated after the
prediction time it serves. The splitter asserts its own invariants on
every call (no train/test index overlap, no [t0, t1] overlap with any
test range, embargo respected); these are runtime assertions, not tests.
"""

from signal_lab.validation import dedupe, labels, periods, point_in_time, splits, timing

__all__ = ["dedupe", "labels", "periods", "point_in_time", "splits", "timing"]
