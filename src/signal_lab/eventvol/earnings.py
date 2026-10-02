"""EVENTVOL earnings calendar: download/cache + event frame build.

Point-in-time contract (frozen in docs/EVENTVOL_PREREGISTRATION.md):
- Source: yfinance ``get_earnings_dates`` per universe ticker.
- Actuals only: rows with non-NaN Reported EPS. Estimate-only rows
  (future scheduled announcements) are excluded.
- t0 mapping: the calendar date of the earnings timestamp; if that date
  is not a trading day, t0 is the next trading day.
- Dedup: one row per (ticker, t0); latest timestamp wins.
- Target: realized_vol_5d = sample std (ddof=1) of daily log returns
  over (t0, t0+5]; partial windows are DROPPED, never filled.
- ``periods.check_no_confirmation`` is enforced on t0 on EVERY build.

The earnings cache (data/eventvol_earnings.parquet) is a local resume
aid only and is never committed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from signal_lab.validation import periods
from signal_lab.vol.prices import DataQualityError

HORIZON_DAYS = 5

EVENT_COLUMNS = [
    "ticker", "t0", "t1", "published_at", "target",
    "eps_estimate", "reported_eps", "surprise_pct", "prev_surprise_pct",
]


def download_earnings(tickers: list[str],
                      cache_path: str | Path,
                      limit: int = 32,
                      force_download: bool = False,
                      log=print) -> pd.DataFrame:
    """Download earnings calendars per ticker; cache to parquet.

    Returns rows: ticker, earnings_ts (tz-aware), eps_estimate,
    reported_eps, surprise_pct. Sequential per-ticker calls (no bulk
    download); failures raise, never silently skipped.
    """
    import yfinance as yf

    cache_path = Path(cache_path)
    if cache_path.exists() and not force_download:
        log(f"[earnings] reusing cached {cache_path}")
        df = pd.read_parquet(cache_path)
        return df

    rows: list[dict] = []
    for i, ticker in enumerate(tickers):
        try:
            ed = yf.Ticker(ticker).get_earnings_dates(limit=limit)
        except Exception as exc:
            raise RuntimeError(
                f"[earnings] download failed for {ticker}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if ed is None or len(ed) == 0:
            log(f"[earnings] {ticker}: no earnings rows returned")
            continue
        for ts, r in ed.iterrows():
            rows.append({
                "ticker": ticker,
                "earnings_ts": pd.Timestamp(ts),
                "eps_estimate": (None if pd.isna(r.get("EPS Estimate"))
                                 else float(r["EPS Estimate"])),
                "reported_eps": (None if pd.isna(r.get("Reported EPS"))
                                 else float(r["Reported EPS"])),
                "surprise_pct": (None if pd.isna(r.get("Surprise(%)"))
                                 else float(r["Surprise(%)"])),
            })
        if (i + 1) % 20 == 0:
            log(f"[earnings] {i + 1}/{len(tickers)} tickers")
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise DataQualityError("[earnings] download produced zero rows")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log(f"[earnings] wrote {cache_path} ({len(df)} rows)")
    return df


def _map_t0(event_date: pd.Timestamp.date,
            trading_ordinals: np.ndarray) -> pd.Timestamp.date:
    """t0 = event date if it is a trading day, else the next trading day."""
    ord_ = event_date.toordinal()
    pos = int(np.searchsorted(trading_ordinals, ord_, side="left"))
    if pos >= len(trading_ordinals):
        raise DataQualityError(
            f"[earnings] event date {event_date} is past the end of the "
            "trading calendar; cannot map t0"
        )
    return pd.Timestamp.fromordinal(int(trading_ordinals[pos])).date()


def build_event_frame(earnings: pd.DataFrame,
                      prices: pd.DataFrame,
                      log=print) -> pd.DataFrame:
    """Build one row per earnings event with the frozen target.

    ``earnings``: output of download_earnings. ``prices``: the vol price
    panel (ticker, date, adj_close, ...). Returns EVENT_COLUMNS.
    """
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    trading_ordinals = np.array(
        sorted(d.date().toordinal() for d in cal), dtype=np.int64)

    # Actuals only, latest timestamp wins per (ticker, date).
    df = earnings.copy()
    df = df[df["reported_eps"].notna()].reset_index(drop=True)
    if len(df) == 0:
        raise DataQualityError("[earnings] zero actual (reported) rows")
    df["event_date"] = pd.to_datetime(df["earnings_ts"]).dt.date
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
        pos = int(np.searchsorted(
            np.array([d.toordinal() for d in dates]), t0.toordinal(),
            side="left"))
        if pos >= len(dates) or dates[pos] != t0:
            raise DataQualityError(
                f"[earnings] t0 {t0} not on the price calendar for {ticker}")
        window = g["adj_close"].to_numpy(dtype=float)[pos:pos + HORIZON_DAYS + 1]
        if len(window) < HORIZON_DAYS + 1:
            continue  # partial label window: drop, never fill
        if not np.all(np.isfinite(window)):
            raise DataQualityError(
                f"[earnings] missing close inside the label window of "
                f"({ticker}, t0={t0}); refusing to label it")
        logc = np.log(window)
        fwd = logc[1:] - logc[:-1]
        target = float(np.std(fwd, ddof=1))
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
    frame = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    if len(frame) == 0:
        raise DataQualityError("[earnings] event frame build produced zero rows")
    frame = frame.drop_duplicates(subset=["ticker", "t0"], keep="last")
    frame = frame.sort_values(["t0", "ticker"]).reset_index(drop=True)
    periods.check_no_confirmation(frame["t0"], "eventvol.build_event_frame")
    log(f"[earnings] event frame: {len(frame)} rows "
        f"(t0 {frame['t0'].min()}..{frame['t0'].max()})")
    return frame


def dev_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Development rows only: t0 <= 2026-06-30 (confirmation-checked)."""
    periods.check_no_confirmation(frame["t0"], "eventvol.dev_frame")
    return frame[periods.in_development(frame["t0"])].reset_index(drop=True)


def test_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Test rows only: 2026-07-01 <= t0 <= 2026-08-31 (confirmation-checked)."""
    periods.check_no_confirmation(frame["t0"], "eventvol.test_frame")
    return frame[periods.in_test(frame["t0"])].reset_index(drop=True)
