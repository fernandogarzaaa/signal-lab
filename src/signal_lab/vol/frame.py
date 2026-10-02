"""VOL frame: one row per (ticker, trading day) with the frozen target.

Columns: ticker, t0 (date), t1 (date = t0 + 5 trading days on the union
calendar), published_at (t0 at 00:00 UTC), target = realized_vol_5d =
sample std (ddof=1) of daily log returns over (t0, t0+5], decimal.

- Rows whose 5-day forward window is partial are DROPPED, never filled.
- Log returns use split/dividend-adjusted closes: r_t = ln(c_t / c_{t-1}).
- ``periods.check_no_confirmation`` is enforced on t0 on EVERY build:
  confirmation rows can never enter a development frame.
- Fail loud: a labeled row whose window contains a NaN close raises
  DataQualityError ("missing closes for labeled rows").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.validation import periods
from signal_lab.vol.prices import DataQualityError

HORIZON_DAYS = 5

FRAME_COLUMNS = ["ticker", "t0", "t1", "published_at", "target"]


def trading_calendar(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Union exchange calendar: sorted unique trading dates in the panel."""
    return pd.DatetimeIndex(sorted(prices["date"].unique()))


def _ticker_frame(dates: np.ndarray, closes: np.ndarray, ticker: str) -> list[dict]:
    """Build rows for one ticker. dates: datetime64[D] sorted; closes: float."""
    n = len(dates)
    if n <= HORIZON_DAYS:
        return []
    logc = np.log(closes)
    # r[i] = log return of day i vs day i-1; r[0] undefined (NaN).
    r = np.empty(n)
    r[0] = np.nan
    r[1:] = logc[1:] - logc[:-1]
    rows = []
    for i in range(n - HORIZON_DAYS):
        window_closes = closes[i:i + HORIZON_DAYS + 1]
        if not np.all(np.isfinite(window_closes)):
            raise DataQualityError(
                f"missing close inside the label window of labeled row "
                f"({ticker}, t0={pd.Timestamp(dates[i]).date().isoformat()}); "
                "refusing to label it"
            )
        fwd = r[i + 1:i + HORIZON_DAYS + 1]
        # fwd is finite here because the 6 closes are finite.
        target = float(np.std(fwd, ddof=1))
        t0 = pd.Timestamp(dates[i]).date()
        t1 = pd.Timestamp(dates[i + HORIZON_DAYS]).date()
        rows.append({
            "ticker": ticker,
            "t0": t0,
            "t1": t1,
            "published_at": pd.Timestamp(t0, tz="UTC"),
            "target": target,
        })
    return rows


def build_frame(prices: pd.DataFrame, horizon: int = HORIZON_DAYS) -> pd.DataFrame:
    """Build the full (ticker, trading day) frame with frozen targets."""
    if horizon != HORIZON_DAYS:
        raise ValueError(f"frozen horizon is {HORIZON_DAYS}, got {horizon}")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    all_rows: list[dict] = []
    for ticker, grp in prices.groupby("ticker"):
        dates = grp["date"].to_numpy().astype("datetime64[D]")
        closes = grp["adj_close"].to_numpy(dtype=float)
        all_rows.extend(_ticker_frame(dates, closes, str(ticker)))
    frame = pd.DataFrame(all_rows, columns=FRAME_COLUMNS)
    if len(frame) == 0:
        raise DataQualityError("frame build produced zero rows")
    periods.check_no_confirmation(frame["t0"], "vol.build_frame")
    return frame


def dev_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Development rows only: t0 <= 2026-06-30 (confirmation-checked)."""
    periods.check_no_confirmation(frame["t0"], "vol.dev_frame")
    return frame[periods.in_development(frame["t0"])].reset_index(drop=True)


def test_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Test rows only: 2026-07-01 <= t0 <= 2026-08-31 (confirmation-checked)."""
    periods.check_no_confirmation(frame["t0"], "vol.test_frame")
    return frame[periods.in_test(frame["t0"])].reset_index(drop=True)


def union_trading_ordinals(prices: pd.DataFrame) -> np.ndarray:
    """Sorted unique trading-date ordinals for the splitter's embargo math."""
    cal = trading_calendar(prices)
    return np.array(sorted(d.date().toordinal() for d in cal), dtype=np.int64)
