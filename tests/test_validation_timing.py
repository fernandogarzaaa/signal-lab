"""Tests for validation.timing: t0/t1 anchoring on real intraday timestamps.

t0 = the last close strictly before the first tradable moment at/after
publication. The news DB carries genuine intraday UTC timestamps (all 24
hours), so the anchoring must handle during-hours, pre-open, after-close,
and non-trading-day publication.

Reference calendar (2026-01): Jan 1 = Thursday (holiday, no prices),
Jan 2 = Friday (trading), Jan 3-4 = weekend, Jan 5 = Monday (trading).
ET = UTC-5 in January.
"""

import numpy as np
import pandas as pd
import pytest

from signal_lab.validation.timing import compute_t0, compute_t1


def _cal():
    # Trading days: 2026-01-02 (Fri) and 2026-01-05 (Mon); Jan 1 holiday.
    return {
        "AAA": np.array(
            [pd.Timestamp("2026-01-02").date(), pd.Timestamp("2026-01-05").date()]
        )
    }


def _pub(ts: str) -> pd.Series:
    return pd.Series(pd.to_datetime([ts], utc=True))


def _t0(ts: str):
    return compute_t0(_pub(ts), _cal(), np.array(["AAA"])).iloc[0]


def _t1(t0_date, horizon=1):
    t0 = pd.Series([t0_date], dtype="object")
    return compute_t1(t0, _cal(), np.array(["AAA"]), horizon).iloc[0]


def test_during_hours_publication_t0_is_pub_date():
    # 15:00 UTC = 10:00 ET, during market hours on Fri 2026-01-02.
    assert _t0("2026-01-02 15:00") == pd.Timestamp("2026-01-02").date()


def test_pre_open_publication_t0_is_previous_trading_day():
    # 12:00 UTC = 07:00 ET, before the open on Fri 2026-01-02.
    # The whole session (including the open reaction) is still ahead,
    # so t0 is the previous trading day... but 2026-01-01 was a holiday
    # with no close, so there is no valid t0 -> None.
    assert _t0("2026-01-02 12:00") is None

    # Same clock time on Mon 2026-01-05: previous trading day 2026-01-02
    # has a close.
    assert _t0("2026-01-05 12:00") == pd.Timestamp("2026-01-02").date()


def test_after_close_publication_t0_is_pub_date():
    # 22:00 UTC = 17:00 ET, after the close on Fri 2026-01-02.
    # The close already happened, so t0 is that day.
    assert _t0("2026-01-02 22:00") == pd.Timestamp("2026-01-02").date()


def test_weekend_publication_t0_is_friday():
    # Saturday 2026-01-03: next trading day is Mon 2026-01-05,
    # last close strictly before it is Fri 2026-01-02.
    assert _t0("2026-01-03 10:00") == pd.Timestamp("2026-01-02").date()


def test_holiday_publication_with_no_prior_close_is_none():
    # Thu 2026-01-01 was a market holiday; no earlier trading day exists
    # in this calendar, so t0 is None (row becomes unlabeled, loudly).
    assert _t0("2026-01-01 15:00") is None


def test_pub_before_first_trading_day_is_none():
    # Published 2025-12-31, before any trading day on the calendar:
    # no "last close strictly before the next trading day".
    assert _t0("2025-12-31 15:00") is None


def test_t1_is_positional_in_trading_days():
    assert _t1(pd.Timestamp("2026-01-02").date(), 1) == pd.Timestamp(
        "2026-01-05"
    ).date()


def test_t1_past_calendar_end_is_none():
    # t0 = last trading day; t0 + 1 trading day does not exist -> None.
    assert _t1(pd.Timestamp("2026-01-05").date(), 1) is None


def test_none_t0_gives_none_t1():
    assert _t1(None, 3) is None


def test_horizon_zero_raises():
    with pytest.raises(ValueError, match="horizon_days"):
        compute_t1(pd.Series([pd.Timestamp("2026-01-02").date()]),
                   _cal(), np.array(["AAA"]), 0)
