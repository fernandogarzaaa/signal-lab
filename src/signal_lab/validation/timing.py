"""Per-sample t0/t1: first tradable time after publication (Phase 1).

Definitions (all in the sample's ticker calendar):

- ``t0``: the trading day whose close *starts* the measurable label window,
  i.e. the last close strictly before the first tradable moment at or
  after publication. Publication timestamps are converted to US/Eastern:
  - Published during market hours (09:30-16:00 ET) on a trading day: the
    session is underway, so the first close-to-close return fully after
    publication starts at that day's close -> t0 is that day.
  - Published pre-open (before 09:30 ET) on a trading day: the whole
    session, including the open reaction, is still ahead -> t0 is the
    previous trading day.
  - Published after the close, or on a non-trading day: the first
    tradable moment is the next trading day's open -> t0 is the last
    trading day strictly before that next trading day.
- ``t1``: t0 plus ``horizon_days`` trading days (positional in the ticker
  calendar). The label window is the half-open (close(t0), close(t1)]
  interval. Articles whose t1 falls past the last available trading day
  get t1 = NaT and are unlabeled (same as the old fwd_days guard).

Worked examples (H = 3, ticker trades Mon-Fri):

- Published Tue 10:00 ET (during hours): t0 = Tue,
  window = (close Tue, close Fri].
- Published Tue 07:00 ET (pre-open): t0 = Mon,
  window = (close Mon, close Thu].
- Published Tue 18:00 ET (after close): t0 = Tue (last trading day before
  Wed's open), window = (close Tue, close Fri].
- Published Sat 12:00 ET: t0 = Fri, window = (close Fri, close Wed].

The old scheme anchored the window at the publication *calendar* date,
which for after-close news measured a first-day return that had already
happened before the news existed, and for pre-open news skipped the
open-reaction day entirely. The t0 rule fixes both while agreeing with
the old scheme for during-hours publication.

No labels, no returns, no future data enter this module: t0/t1 are pure
functions of (publication timestamp, ticker calendar, horizon).
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

MARKET_CLOSE_ET = time(16, 0)
MARKET_OPEN_ET = time(9, 30)


def _to_et(ts: pd.Series) -> pd.Series:
    """UTC timestamps -> US/Eastern (naive)."""
    return pd.to_datetime(ts, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)


def _ordinals(cal: np.ndarray) -> np.ndarray:
    """Trading calendar as sorted int ordinals (searchsorted-safe)."""
    return np.array([pd.Timestamp(d).date().toordinal() for d in cal], dtype=np.int64)


def compute_t0(
    published_at: pd.Series,
    calendars: dict[str, np.ndarray],
    tickers: np.ndarray,
) -> pd.Series:
    """Per-article t0 date (see module docstring).

    ``calendars``: ticker -> sorted array of trading ``datetime.date``.
    ``tickers``: per-article ticker, aligned with ``published_at``.
    Returns a Series of ``datetime.date`` (None where the ticker has no
    calendar or no qualifying trading day exists).
    """
    pub_et = _to_et(published_at)
    pub_ord = pub_et.dt.date.apply(lambda d: d.toordinal()).to_numpy()
    pub_times = pub_et.dt.time.to_numpy()
    t0 = np.full(len(published_at), None, dtype=object)
    for ticker, cal in calendars.items():
        m = np.flatnonzero(tickers == ticker)
        if len(m) == 0 or len(cal) == 0:
            continue
        cord = _ordinals(cal)
        cal_set = set(cord.tolist())
        d = pub_ord[m]
        t = pub_times[m]
        on_trading_day = np.array([x in cal_set for x in d])
        during = on_trading_day & (t >= MARKET_OPEN_ET) & (t <= MARKET_CLOSE_ET)
        preopen = on_trading_day & (t < MARKET_OPEN_ET)
        rest = ~(during | preopen)

        t0_ord = np.full(len(m), -1, dtype=np.int64)
        # During-hours publication on a trading day: t0 is that day.
        t0_ord[during] = d[during]
        # Pre-open on a trading day: the whole session (incl. the open
        # reaction) is still ahead, so t0 is the previous trading day.
        idx = np.searchsorted(cord, d[preopen], side="left")
        ok = idx > 0
        t0_ord[np.flatnonzero(preopen)[ok]] = cord[idx[ok] - 1]
        # After close, or on a non-trading day: t0 is the last trading day
        # strictly before the next trading day strictly after the pub date
        # (= the close the next open follows).
        nxt = np.searchsorted(cord, d[rest], side="right")
        ok2 = nxt > 0
        t0_ord[np.flatnonzero(rest)[ok2]] = cord[nxt[ok2] - 1]

        vals = np.full(len(m), None, dtype=object)
        good = t0_ord >= 0
        vals[good] = [
            pd.Timestamp.fromordinal(int(o)).date() for o in t0_ord[good]
        ]
        t0[m] = vals
    return pd.Series(t0, index=published_at.index)


def compute_t1(
    t0: pd.Series,
    calendars: dict[str, np.ndarray],
    tickers: np.ndarray,
    horizon_days: int,
) -> pd.Series:
    """t0 + ``horizon_days`` trading days in the ticker's calendar.

    pd.NaT where t0 is missing or the horizon runs past the last trading
    day (article too recent to label).
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    t1 = np.full(len(t0), None, dtype=object)
    t0_ord = t0.apply(
        lambda d: d.toordinal() if d is not None and not pd.isna(d) else -1
    ).to_numpy(dtype=np.int64)
    for ticker, cal in calendars.items():
        m = np.flatnonzero(tickers == ticker)
        if len(m) == 0 or len(cal) == 0:
            continue
        cord = _ordinals(cal)
        mv = m[t0_ord[m] >= 0]
        if len(mv) == 0:
            continue
        pos = np.searchsorted(cord, t0_ord[mv], side="left")
        exact = (pos < len(cord)) & (cord[np.minimum(pos, len(cord) - 1)] == t0_ord[mv])
        dest = pos + horizon_days
        ok = exact & (dest < len(cord))
        t1[mv[ok]] = [
            pd.Timestamp.fromordinal(int(o)).date() for o in cord[dest[ok]]
        ]
    return pd.Series(t1, index=t0.index)
