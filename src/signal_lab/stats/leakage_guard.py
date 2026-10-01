"""Fail-loud baseline/event-window leakage guard.

Every event-study style computation in this repo has a baseline
(estimation) window and an event window. If any baseline observation sits
on or after the event-window start, the baseline is contaminated by the
event itself: CAR standardization divides by event-inflated volatility,
abnormal z-scores absorb the move they are supposed to flag, and every
p-value downstream is a lie.

This module provides one explicit, fail-loud check (pattern from
jiamingpan/agent-quant-research's ``_assert_no_baseline_leakage``,
reimplemented in our typed style): the baseline labels must be strictly
before the event-window start. It raises ``BaselineLeakageError`` instead
of using a bare ``assert`` so the check survives ``python -O``.

Wired into:
- ``signal_lab.stats.car_tests.event_ar_panel`` (estimation window vs
  event window, checked per event on the finished panel)
- ``signal_lab.stats.car_tests._scar_frame`` (any hand-built panel is
  re-checked before the estimation/event split, defense in depth)
- ``signal_lab.explain_move._price_block`` (trailing 60-day baseline for
  the abnormal z-score must end before the event day)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


class BaselineLeakageError(AssertionError):
    """Raised when a baseline (estimation) window reaches on or after the
    event-window start. This is never a soft warning: a contaminated
    baseline silently invalidates the study, so the computation dies here."""


def assert_baseline_precedes_event_window(
    baseline_labels: Sequence[Any] | Iterable[Any],
    event_window_start: Any,
    *,
    context: str = "",
) -> None:
    """Fail loudly unless every baseline label is strictly before the
    event-window start.

    baseline_labels: the baseline window's dates or trading-day offsets,
    in any order. event_window_start: the first date/offset of the event
    window. Both must be mutually comparable (dates with dates, ints with
    ints). An empty baseline passes vacuously; downstream minimum-size
    checks still apply.

    Raises BaselineLeakageError listing the leaking labels.
    """
    leaking = [b for b in baseline_labels if b >= event_window_start]
    if leaking:
        where = f" [{context}]" if context else ""

        def _fmt(label: Any) -> str:
            iso = getattr(label, "isoformat", None)
            if callable(iso):
                try:
                    return str(iso())
                except Exception:
                    pass
            return repr(label)

        raise BaselineLeakageError(
            "Baseline leakage: baseline observations must be strictly before "
            f"the event-window start {_fmt(event_window_start)}{where}; got "
            f"{len(leaking)} leaking label(s): "
            f"{[_fmt(b) for b in leaking[:8]]}"
            + (" ..." if len(leaking) > 8 else "")
        )
