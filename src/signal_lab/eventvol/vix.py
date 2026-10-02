"""EVENTVOL VIX data: ^VIX / ^VIX3M download, cache, point-in-time lookup.

VIX history is the market-implied leg of the iv_proxy baseline and the
source of the VIX features. Verified reachable from yfinance on
2026-10-02 (see docs/EVENTVOL_PREREGISTRATION.md STEP 0).

Point-in-time contract: every lookup uses closes on trading days <= t0
only. A missing VIX close at t0 falls back to the most recent prior
trading day's close and is counted by the caller.

The cache (data/eventvol_vix.parquet) is a local resume aid only and is
never committed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VIX_SYMBOL = "^VIX"
VIX3M_SYMBOL = "^VIX3M"
VIX_START = "2022-01-01"
VIX_END = "2026-07-15"
TRADING_DAYS_PER_YEAR = 252


def download_vix(cache_path: str | Path,
                 force_download: bool = False,
                 log=print) -> pd.DataFrame:
    """Download ^VIX and ^VIX3M daily closes; cache to parquet.

    Returns columns: date (datetime64), vix, vix3m. Raises on download
    failure or empty result; never silently backfills.
    """
    import yfinance as yf

    cache_path = Path(cache_path)
    if cache_path.exists() and not force_download:
        log(f"[vix] reusing cached {cache_path}")
        df = pd.read_parquet(cache_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    frames = {}
    for sym, col in ((VIX_SYMBOL, "vix"), (VIX3M_SYMBOL, "vix3m")):
        try:
            d = yf.download(sym, start=VIX_START, end=VIX_END,
                            progress=False, auto_adjust=False)
        except Exception as exc:
            raise RuntimeError(
                f"[vix] download failed for {sym}: "
                f"{type(exc).__name__}: {exc}") from exc
        if d is None or len(d) == 0:
            raise RuntimeError(f"[vix] download for {sym} returned zero rows")
        close = d["Close"]
        if isinstance(close, pd.DataFrame):  # multi-ticker column layout
            close = close.iloc[:, 0]
        s = pd.to_numeric(close, errors="coerce")
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        frames[col] = s
    df = pd.DataFrame({"date": frames["vix"].index, "vix": frames["vix"].to_numpy()})
    vix3m = frames["vix3m"]
    df["vix3m"] = vix3m.reindex(df["date"]).to_numpy()
    if df["vix"].isna().any():
        raise RuntimeError("[vix] NaN VIX closes after download; refusing")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log(f"[vix] wrote {cache_path} ({len(df)} rows)")
    return df


def vix_at(t0, vix: pd.DataFrame) -> tuple[float, float, bool]:
    """(vix_close, vix3m_close, used_fallback) at t0, point-in-time.

    Uses the close on the latest trading day <= t0. ``used_fallback`` is
    True when t0 itself has no VIX row (e.g. a holiday mapping edge).
    """
    t0d = pd.Timestamp(t0).normalize()
    dates = vix["date"].to_numpy()
    ords = np.array([pd.Timestamp(d).toordinal() for d in dates])
    pos = int(np.searchsorted(ords, t0d.toordinal(), side="right")) - 1
    if pos < 0:
        raise ValueError(f"[vix] t0 {t0d.date()} predates VIX history")
    row = vix.iloc[pos]
    return float(row["vix"]), float(row["vix3m"]), bool(ords[pos] != t0d.toordinal())


def vix_features_at(t0, vix: pd.DataFrame) -> dict[str, float]:
    """Point-in-time VIX features as of t0's close.

    vix_level      = VIX(t0)/100/sqrt(252) (decimal daily)
    vix_5d_change  = VIX(t0) - VIX(t0 - 5 trading days), in VIX points
    vix_runup_22d  = VIX(t0) - mean(VIX trailing 22 trading days)
    vix_slope      = VIX3M(t0)/VIX(t0) - 1 (term-structure slope)
    """
    t0d = pd.Timestamp(t0).normalize()
    dates = vix["date"].to_numpy()
    ords = np.array([pd.Timestamp(d).toordinal() for d in dates])
    pos = int(np.searchsorted(ords, t0d.toordinal(), side="right")) - 1
    if pos < 21:
        return {k: np.nan for k in
                ("vix_level", "vix_5d_change", "vix_runup_22d", "vix_slope")}
    closes = vix["vix"].to_numpy(dtype=float)
    closes3m = vix["vix3m"].to_numpy(dtype=float)
    c0, c3m = closes[pos], closes3m[pos]
    if not (np.isfinite(c0) and c0 > 0 and np.isfinite(c3m) and c3m > 0):
        return {k: np.nan for k in
                ("vix_level", "vix_5d_change", "vix_runup_22d", "vix_slope")}
    trail22 = closes[pos - 21:pos + 1]
    if not np.all(np.isfinite(trail22)):
        return {k: np.nan for k in
                ("vix_level", "vix_5d_change", "vix_runup_22d", "vix_slope")}
    return {
        "vix_level": float(c0 / 100.0 / np.sqrt(TRADING_DAYS_PER_YEAR)),
        "vix_5d_change": float(c0 - closes[pos - 5]),
        "vix_runup_22d": float(c0 - trail22.mean()),
        "vix_slope": float(c3m / c0 - 1.0),
    }


VIX_FEATURE_COLUMNS: list[str] = [
    "vix_level", "vix_5d_change", "vix_runup_22d", "vix_slope",
]
