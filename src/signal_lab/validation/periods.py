"""Frozen research periods (Phase 1).

The 0.6.0 research program fixes three evaluation periods by t0 date.
These are constants, not rolling windows: moving them after seeing
results would be test-set peeking. New data ingested after 2026-09-30
accrues to the confirmation / live-log regime (Phase 6), never back into
development or test.

- DEVELOPMENT: t0 <= 2026-06-30. Walk-forward validation and ALL model /
  feature / threshold selection happen here and only here.
- TEST: 2026-07-01 <= t0 <= 2026-08-31. Untouched future test. Evaluated
  exactly once per finalized configuration (Phase 3 gate confirmation).
  Never used for selection, tuning, thresholds, or calibration.
- CONFIRMATION: t0 >= 2026-09-01. Frozen. Never used in development, not
  even for "a quick look". The Phase 6 live log starts here.

The label horizon is 3 trading days and prices run through 2026-09-30,
so the last labelable t0 is ~2026-09-25; the confirmation period's
outcomes become observable in early October 2026.

``check_no_confirmation`` is called by every development entry point; it
raises ``ConfirmationLeakError`` if any row falls in the confirmation
period, so the freeze is enforced in code, not just in prose.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

DEV_END: date = date(2026, 6, 30)
TEST_START: date = date(2026, 7, 1)
TEST_END: date = date(2026, 8, 31)
CONFIRMATION_START: date = date(2026, 9, 1)


class ConfirmationLeakError(RuntimeError):
    """Raised when confirmation-period data enters a development path."""


def _as_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.date


def in_development(t0: pd.Series) -> pd.Series:
    return _as_date(t0) <= DEV_END


def in_test(t0: pd.Series) -> pd.Series:
    d = _as_date(t0)
    return (d >= TEST_START) & (d <= TEST_END)


def in_confirmation(t0: pd.Series) -> pd.Series:
    return _as_date(t0) >= CONFIRMATION_START


def check_no_confirmation(t0: pd.Series, context: str) -> None:
    """Raise if any sample falls in the frozen confirmation period."""
    bad = in_confirmation(t0)
    if bool(bad.any()):
        n = int(bad.sum())
        raise ConfirmationLeakError(
            f"[{context}] {n} sample(s) fall in the frozen confirmation "
            f"period (t0 >= {CONFIRMATION_START}); confirmation data must "
            "never enter development paths."
        )


def describe() -> dict:
    """JSON-safe description of the periods (for provenance blocks)."""
    return {
        "development": {"t0_end": DEV_END.isoformat(), "use": "walk-forward validation; all selection"},
        "test": {
            "t0_start": TEST_START.isoformat(),
            "t0_end": TEST_END.isoformat(),
            "use": "single untouched future evaluation; no selection/tuning/calibration",
        },
        "confirmation": {
            "t0_start": CONFIRMATION_START.isoformat(),
            "use": "frozen; Phase 6 live log only",
        },
    }
