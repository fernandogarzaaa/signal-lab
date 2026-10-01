"""Tests for the fail-loud baseline/event-window leakage guard.

Covers the guard unit itself, its wiring into the event-study path
(event_ar_panel construction + cross_sectional_tests split), and the
explainer's abnormal-z trailing baseline. Hostile inputs must raise
BaselineLeakageError; clean inputs must pass untouched.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from signal_lab.stats import car_tests
from signal_lab.stats.leakage_guard import (
    BaselineLeakageError,
    assert_baseline_precedes_event_window,
)


# ---------------------------------------------------------------------------
# Guard unit
# ---------------------------------------------------------------------------

def test_clean_dates_pass():
    base = [date(2026, 1, d) for d in range(5, 10)]
    assert_baseline_precedes_event_window(base, date(2026, 1, 10))  # no raise


def test_clean_offsets_pass():
    assert_baseline_precedes_event_window(list(range(-120, -1)), -1)


def test_empty_baseline_passes_vacuously():
    assert_baseline_precedes_event_window([], date(2026, 1, 10))


def test_boundary_leak_raises():
    # A baseline observation ON the event-window start is leakage.
    base = [date(2026, 1, 8), date(2026, 1, 9), date(2026, 1, 10)]
    with pytest.raises(BaselineLeakageError):
        assert_baseline_precedes_event_window(base, date(2026, 1, 10))


def test_post_start_leak_raises():
    base = [date(2026, 1, 8), date(2026, 1, 12)]
    with pytest.raises(BaselineLeakageError):
        assert_baseline_precedes_event_window(base, date(2026, 1, 10))


def test_offset_leak_raises():
    with pytest.raises(BaselineLeakageError):
        assert_baseline_precedes_event_window([-3, -2, -1, 0], -1)


def test_error_message_names_leaking_labels_and_context():
    base = [date(2026, 1, 9), date(2026, 1, 10)]
    with pytest.raises(BaselineLeakageError) as exc:
        assert_baseline_precedes_event_window(
            base, date(2026, 1, 10), context="unit-check")
    msg = str(exc.value)
    assert "2026-01-10" in msg
    assert "unit-check" in msg


# ---------------------------------------------------------------------------
# Event-study wiring
# ---------------------------------------------------------------------------

def _prices(n_days: int = 260, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    frames = []
    for i, t in enumerate(("AAA", "SPY")):
        # Different noise per series so ARs are not degenerate zeros.
        rets = rng.normal(0.0005, 0.01, size=n_days) + i * 0.0001
        closes = 100.0 * np.exp(np.cumsum(rets))
        frames.append(pd.DataFrame({
            "ticker": t, "date": [d.date() for d in dates], "close": closes,
        }))
    return pd.concat(frames, ignore_index=True)


def test_event_ar_panel_clean_construction_passes_guard():
    events = pd.DataFrame({
        "ticker": ["AAA"], "event_date": [date(2025, 8, 1)], "sentiment": [0.9],
    })
    panel = car_tests.event_ar_panel(events, _prices(), pre=1, post=1)
    assert not panel.empty
    # Estimation offsets are strictly before the event-window start (-1).
    assert panel.loc[~panel["in_event"], "offset"].max() < -1


def _hostile_panel() -> pd.DataFrame:
    """One event where the same trading day (offset 0, the event-window
    start region) sits in BOTH the estimation window and the event window.
    Simulates a corrupted join or a bad refactor of the window logic."""
    rows = []
    for offset in range(-120, 3):  # event window [-1, +1]
        if offset == 0:
            in_event = False  # HOSTILE: estimation reaches the event window
        else:
            in_event = -1 <= offset <= 1
        rows.append({
            "event_id": 0, "ticker": "AAA", "event_date": date(2025, 8, 1),
            "offset": offset, "ar": 0.001, "in_event": in_event,
        })
    return pd.DataFrame(rows)


def test_hostile_panel_fires_in_cross_sectional_tests():
    with pytest.raises(BaselineLeakageError):
        car_tests.cross_sectional_tests(_hostile_panel(), pre=1, post=1)


def test_clean_panel_passes_cross_sectional_tests():
    events = pd.DataFrame({
        "ticker": ["AAA"] * 4,
        # est_days=120 needs >= 121 trading days before each event.
        "event_date": [date(2025, 9, 2), date(2025, 10, 1),
                       date(2025, 11, 3), date(2025, 12, 1)],
        "sentiment": [0.9] * 4,
    })
    panel = car_tests.event_ar_panel(events, _prices(n_days=300), pre=1, post=1)
    assert not panel.empty
    res = car_tests.cross_sectional_tests(panel, pre=1, post=1)
    assert res["n_events"] >= 3


# ---------------------------------------------------------------------------
# Explainer wiring (same guard, trailing-baseline call shape)
# ---------------------------------------------------------------------------

def test_explainer_baseline_shape_clean_passes():
    event_day = pd.Timestamp("2026-04-24")
    hist_dates = pd.bdate_range("2026-01-26", periods=60).tolist()
    assert_baseline_precedes_event_window(
        hist_dates, event_day,
        context="abnormal_z baseline for AAA@2026-04-24")


def test_explainer_baseline_shape_hostile_fires():
    # A refactor that lets the trailing baseline touch the event day.
    event_day = pd.Timestamp("2026-04-24")
    hist_dates = pd.bdate_range(end="2026-04-24", periods=60).tolist()
    assert hist_dates[-1] == event_day
    with pytest.raises(BaselineLeakageError):
        assert_baseline_precedes_event_window(
            hist_dates, event_day,
            context="abnormal_z baseline for AAA@2026-04-24")
