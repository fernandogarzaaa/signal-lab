"""TAILQ event frame: one row per earnings event with the 3-day event return target.

Reuses the frozen EVENTVOL earnings pipeline for the point-in-time
event definition (yfinance get_earnings_dates, actuals only, latest
timestamp wins per (ticker, date), t0 = earnings date mapped to the
next trading day). The TAILQ target differs: the 3-day event return
(sum of daily log returns over (t0, t0+3]) instead of realized_vol_5d.

Point-in-time contract (frozen in docs/TAILQ_PREREGISTRATION.md):
- t0 = calendar date of the earnings timestamp, mapped to the next
  trading day via the price calendar.
- Target r_3d = sum of daily log returns over (t0, t0+3]
  (t0+1, t0+2, t0+3). Partial windows are DROPPED, never filled.
- Forecast origin is t0's close: features may use returns/prices
  through t0 only; the label window never enters a feature.
- periods.check_no_confirmation enforced on t0 on EVERY build.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from signal_lab.eventvol.earnings import download_earnings
from signal_lab.validation import periods
from signal_lab.vol.prices import DataQualityError

HORIZON_DAYS = 3

TAILQ_EVENT_COLUMNS = [
    "ticker", "t0", "t1", "published_at", "target",
    "eps_estimate", "reported_eps", "surprise_pct", "prev_surprise_pct",
]

EARNINGS_CACHE = "data/tailq_earnings.parquet"


def _map_t0(event_date, trading_ordinals: np.ndarray):
    """t0 = event date if it is a trading day, else the next trading day."""
    ord_ = pd.Timestamp(event_date).date().toordinal()
    pos = int(np.searchsorted(trading_ordinals, ord_, side="left"))
    if pos >= len(trading_ordinals):
        raise DataQualityError(
            f"[tailq] event date {event_date} is past the end of the "
            "trading calendar; cannot map t0"
        )
    return pd.Timestamp.fromordinal(int(trading_ordinals[pos])).date()


def build_tailq_frame(earnings: pd.DataFrame,
                      prices: pd.DataFrame,
                      log=print) -> pd.DataFrame:
    """Build one row per earnings event with the 3-day event-return target.

    ``earnings``: output of eventvol.earnings.download_earnings.
    ``prices``: the vol price panel (ticker, date, adj_close, ...).
    Returns TAILQ_EVENT_COLUMNS.
    """
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(prices["date"]).unique()))
    trading_ordinals = np.array(
        sorted(d.date().toordinal() for d in cal), dtype=np.int64)
    last_trading_date = cal.date.max()

    # Actuals only, latest timestamp wins per (ticker, date).
    df = earnings.copy()
    df = df[df["reported_eps"].notna()].reset_index(drop=True)
    if len(df) == 0:
        raise DataQualityError("[tailq] zero actual (reported) rows")
    df["event_date"] = pd.to_datetime(df["earnings_ts"]).dt.date
    past_end = df["event_date"] > last_trading_date
    if past_end.any():
        log(f"[tailq] dropped {int(past_end.sum())} events past the "
            f"price calendar end ({last_trading_date})")
        df = df[~past_end].reset_index(drop=True)
    df = df.sort_values(["ticker", "event_date", "earnings_ts"])
    df = df.drop_duplicates(subset=["ticker", "event_date"], keep="last")

    # Per-ticker previous-quarter surprise (knowable at t0).
    df = df.sort_values(["ticker", "earnings_ts"]).reset_index(drop=True)
    df["prev_surprise_pct"] = (
        df.groupby("ticker")["surprise_pct"].shift(1) / 100.0)

    closes: dict[str, pd.DataFrame] = {
        str(t): g.sort_values("date").reset_index(drop=True)
        for t, g in prices.groupby("ticker")
    }

    rows: list[dict] = []
    for _, e in df.iterrows():
        ticker = str(e["ticker"])
        t0 = _map_t0(e["event_date"], trading_ordinals)
        g = closes.get(ticker)
        if g is None:
            continue
        dates = pd.to_datetime(g["date"]).dt.date.to_numpy()
        date_ords = np.array([d.toordinal() for d in dates])
        pos = int(np.searchsorted(date_ords, t0.toordinal(), side="left"))
        if pos >= len(dates) or dates[pos] != t0:
            raise DataQualityError(
                f"[tailq] t0 {t0} not on the price calendar for {ticker}")
        window = g["adj_close"].to_numpy(dtype=float)[pos:pos + HORIZON_DAYS + 1]
        if len(window) < HORIZON_DAYS + 1:
            continue  # partial label window: drop, never fill
        if not np.all(np.isfinite(window)):
            raise DataQualityError(
                f"[tailq] missing close inside the label window of "
                f"({ticker}, t0={t0}); refusing to label it")
        logc = np.log(window)
        fwd = logc[1:] - logc[:-1]
        target = float(np.sum(fwd))  # 3-day event return, decimal
        t1 = dates[pos + HORIZON_DAYS]
        rows.append({
            "ticker": ticker,
            "t0": t0,
            "t1": t1,
            "published_at": pd.Timestamp(t0, tz="UTC"),
            "target": target,
            "eps_estimate": e["eps_estimate"],
            "reported_eps": e["reported_eps"],
            "surprise_pct": e["surprise_pct"],
            "prev_surprise_pct": e["prev_surprise_pct"],
        })
    frame = pd.DataFrame(rows, columns=TAILQ_EVENT_COLUMNS)
    if len(frame) == 0:
        raise DataQualityError("[tailq] event frame build produced zero rows")
    frame = frame.drop_duplicates(subset=["ticker", "t0"], keep="last")
    frame = frame.sort_values(["t0", "ticker"]).reset_index(drop=True)
    periods.check_no_confirmation(frame["t0"], "tailq.build_tailq_frame")
    log(f"[tailq] event frame: {len(frame)} rows "
        f"(t0 {frame['t0'].min()}..{frame['t0'].max()})")
    return frame


def dev_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Development rows only: t0 <= 2026-06-30 (confirmation-checked)."""
    periods.check_no_confirmation(frame["t0"], "tailq.dev_frame")
    return frame[periods.in_development(frame["t0"])].reset_index(drop=True)


def test_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Test rows only: 2026-07-01 <= t0 <= 2026-08-31 (confirmation-checked)."""
    periods.check_no_confirmation(frame["t0"], "tailq.test_frame")
    return frame[periods.in_test(frame["t0"])].reset_index(drop=True)


def load_earnings(tickers: list[str],
                  cache_path: str | Path = EARNINGS_CACHE,
                  force_download: bool = False,
                  log=print) -> pd.DataFrame:
    """Earnings download with local parquet resume cache (never committed)."""
    return download_earnings(tickers, cache_path,
                             force_download=force_download, log=log)
