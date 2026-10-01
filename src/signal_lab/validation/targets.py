"""Versioned forward-return targets (Phase 2).

Target set version 2.0. For each article, anchored at its t0 (the first
tradable time after publication, computed by
``signal_lab.validation.timing`` -- t0 is an INPUT here, never
recomputed), the label window is the half-open (close(t0), close(t0+H)]
interval for each horizon H in ``TargetConfig.horizons``:

- ``ret_Hd``: the ticker's close-to-close return over the window.
- ``mkt_Hd``: the benchmark's close-to-close return over the same window.
- ``excess_Hd``: ``ret_Hd - mkt_Hd`` (abnormal return vs the benchmark).
- ``direction_Hd``: 1.0 iff ``ret_Hd > 0`` else 0.0 (NaN when the window
  is partial -- a fabricated 0 would be a made-up label).
- ``realized_vol_5d``: sample std (ddof=1) of the ticker's daily LOG
  returns over the five trading days (t0, t0+5].

``excess_3d`` is formula-identical to the Phase 1 label input ``abn_ret``
(``models.labels.build_labels`` with window_days=3, benchmark SPY): the
forward-return arithmetic below mirrors
``models.labels.forward_cumulative_abnormal`` exactly (same daily simple
returns, same benchmark-else-universe-mean market leg, same partial-day
counting), and ``tests/test_targets_features.py`` pins the equality on
both synthetic and real data. The v2 target set therefore subsumes the
Phase 1 label input; the per-fold top-decile cutoff that turns
``abn_ret``/``excess_3d`` into the primary classification label stays in
``signal_lab.validation.labels`` (it is a train-sample statistic and must
be computed on train data only).

Partial windows: when fewer than H forward trading days carry both ticker
and market returns, that horizon's four columns are NaN. The row is kept
(other horizons may be complete). Insufficient forward history is never
filled or forward-filled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_SET_VERSION = "2.0"

#: Forward window (in trading days) for the realized-volatility target.
REALIZED_VOL_WINDOW = 5


@dataclass(frozen=True)
class TargetConfig:
    """Parameters of the versioned target set."""

    horizons: tuple[int, ...] = (1, 3, 5)
    benchmark: str = "SPY"

    def validated(self) -> TargetConfig:
        horizons = tuple(self.horizons)
        if not horizons:
            raise ValueError("horizons must be a non-empty tuple of positive ints")
        for h in horizons:
            if isinstance(h, bool) or not isinstance(h, int) or h < 1:
                raise ValueError(
                    f"horizons must be positive ints, got {h!r}"
                )
        if len(set(horizons)) != len(horizons):
            raise ValueError(f"horizons must be unique, got {horizons!r}")
        if not isinstance(self.benchmark, str) or not self.benchmark:
            raise ValueError("benchmark must be a non-empty string")
        if horizons != self.horizons:
            return TargetConfig(horizons=horizons, benchmark=self.benchmark)
        return self


def target_column_names(config: TargetConfig) -> list[str]:
    """Output column names for a target config, in build order."""
    cols = ["target_version"]
    for h in config.horizons:
        cols += [f"ret_{h}d", f"mkt_{h}d", f"excess_{h}d", f"direction_{h}d"]
    cols.append("realized_vol_5d")
    return cols


def _clean_close_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """ticker/date/close, date-sorted, deduplicated (one row per day)."""
    p = prices[["ticker", "date", "close"]].copy()
    p["date"] = pd.to_datetime(p["date"]).dt.date
    p = p.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])
    return p.reset_index(drop=True)


def _market_daily_returns(rets: pd.DataFrame, benchmark: str) -> pd.Series:
    """Per-date market simple return: benchmark, else universe equal-weight
    mean. Mirrors ``signal_lab.models.labels._market_daily_returns`` exactly
    so that excess_Hd stays formula-identical to abn_ret."""
    bench = (
        rets[rets["ticker"] == benchmark]
        .drop_duplicates("date")
        .set_index("date")["ret"]
    )
    umean = rets[rets["ticker"] != benchmark].groupby("date")["ret"].mean()
    dates = rets["date"].drop_duplicates().sort_values()
    mkt = bench.reindex(dates)
    return mkt.fillna(umean.reindex(dates)).rename("mkt_ret")


def _forward_horizon_frame(
    prices: pd.DataFrame, horizons: tuple[int, ...], benchmark: str
) -> pd.DataFrame:
    """Per (ticker, date): forward cumulative returns for each horizon.

    Mirrors ``signal_lab.models.labels.forward_cumulative_abnormal``: the
    window for horizon H anchored at date D is (close(D), close(D+H)],
    built from daily simple returns; a forward day counts only when both
    the ticker and the market leg have a return.
    """
    p = _clean_close_frame(prices)
    p["ret"] = p.groupby("ticker")["close"].pct_change()
    p = p.merge(
        _market_daily_returns(p, benchmark), left_on="date", right_index=True,
        how="left",
    )
    out = p[["ticker", "date"]].copy()
    for h in horizons:
        cum_t = pd.Series(1.0, index=p.index)
        cum_m = pd.Series(1.0, index=p.index)
        fwd_days = pd.Series(0, index=p.index)
        for k in range(1, h + 1):
            rt = p.groupby("ticker")["ret"].shift(-k)
            rm = p.groupby("ticker")["mkt_ret"].shift(-k)
            ok = rt.notna() & rm.notna()
            cum_t[ok] = cum_t[ok] * (1.0 + rt[ok])
            cum_m[ok] = cum_m[ok] * (1.0 + rm[ok])
            fwd_days = fwd_days + ok.astype(int)
        out[f"ret_{h}d"] = cum_t - 1.0
        out[f"mkt_{h}d"] = cum_m - 1.0
        out[f"excess_{h}d"] = out[f"ret_{h}d"] - out[f"mkt_{h}d"]
        r = out[f"ret_{h}d"]
        out[f"direction_{h}d"] = np.where(
            r.isna(), np.nan, np.where(r > 0, 1.0, 0.0)
        )
        out[f"_fwd_days_{h}d"] = fwd_days
    return out


def _forward_realized_vol(
    prices: pd.DataFrame,
    tickers: np.ndarray,
    t0: pd.Series,
    window: int = REALIZED_VOL_WINDOW,
) -> np.ndarray:
    """Sample std (ddof=1) of daily log returns over (t0, t0+window].

    The window holds the ``window`` trading days strictly after t0; all of
    them must exist with finite log returns, otherwise NaN.
    """
    p = _clean_close_frame(prices)
    out = np.full(len(tickers), np.nan)
    t0o = (
        pd.to_datetime(t0, errors="coerce")
        .map(lambda d: d.toordinal() if pd.notna(d) else -1)
        .to_numpy(dtype=np.int64)
    )
    for ticker, sub in p.groupby("ticker"):
        idx = np.flatnonzero(tickers == ticker)
        if len(idx) == 0:
            continue
        sub = sub.sort_values("date")
        cord = np.array([d.toordinal() for d in sub["date"]], dtype=np.int64)
        c = sub["close"].to_numpy(dtype=float)
        lrf = np.full(len(c), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(c[1:] / c[:-1])
        lr[~np.isfinite(lr)] = np.nan
        lrf[1:] = lr
        # A log return is only valid between two positive closes.
        bad = np.zeros(len(c), dtype=bool)
        bad[0] = True
        bad[1:] = (c[1:] <= 0) | (c[:-1] <= 0)
        lrf[bad] = np.nan
        pos = np.searchsorted(cord, t0o[idx], side="left")
        in_cal = pos < len(cord)
        exact = np.zeros(len(idx), dtype=bool)
        exact[in_cal] = cord[pos[in_cal]] == t0o[idx[in_cal]]
        exact &= t0o[idx] >= 0
        for ii, pp in zip(idx[exact], pos[exact]):
            if pp + window < len(cord):
                w = lrf[pp + 1 : pp + 1 + window]
                if np.isfinite(w).all():
                    out[ii] = float(np.std(w, ddof=1))
    return out


def build_targets(
    news: pd.DataFrame,
    prices: pd.DataFrame,
    config: TargetConfig,
    log=print,
) -> pd.DataFrame:
    """Compute the versioned target set for each news row.

    ``news`` needs columns ``ticker`` and ``t0`` (date; the first tradable
    time after publication -- an input, never recomputed here).
    ``prices`` needs columns ``ticker, date, open, high, low, close,
    volume`` (only ``close`` is used; the rest are required so a caller
    cannot silently pass a close-only frame and get silently different
    targets later).

    Returns a DataFrame on the input index with ``target_version`` plus
    ``target_column_names(config)``. Rows whose t0 is missing or not on
    the ticker's calendar, or whose forward window is partial, get NaN
    for the affected horizon's columns (the row is kept).
    """
    config = config.validated()
    missing_news = {"ticker", "t0"} - set(news.columns)
    if missing_news:
        raise ValueError(f"[targets] news missing columns {missing_news}")
    missing_px = {"ticker", "date", "open", "high", "low", "close", "volume"} - set(
        prices.columns
    )
    if missing_px:
        raise ValueError(f"[targets] prices missing columns {missing_px}")

    fwd = _forward_horizon_frame(prices, config.horizons, config.benchmark)
    t0d = pd.to_datetime(news["t0"], errors="coerce").dt.date

    out = pd.DataFrame(index=news.index)
    out["target_version"] = TARGET_SET_VERSION
    keyed = news[["ticker"]].copy()
    keyed["_t0d"] = t0d
    m = keyed.merge(
        fwd, left_on=["ticker", "_t0d"], right_on=["ticker", "date"], how="left"
    )
    if len(m) != len(news):
        raise RuntimeError(
            f"[targets] merge changed row count {len(news)} -> {len(m)}"
        )
    for h in config.horizons:
        cols = [f"ret_{h}d", f"mkt_{h}d", f"excess_{h}d", f"direction_{h}d"]
        fwdays = m[f"_fwd_days_{h}d"]
        partial = fwdays.isna() | (fwdays < h)
        n_partial = int(partial.sum())
        if n_partial:
            log(
                f"[targets] horizon {h}d: {n_partial} rows with partial "
                "forward windows -> NaN (rows kept)"
            )
        m.loc[partial, cols] = np.nan
        for c in cols:
            out[c] = m[c].to_numpy()
    out["realized_vol_5d"] = _forward_realized_vol(
        prices, news["ticker"].to_numpy(), news["t0"]
    )
    n_novol = int(out["realized_vol_5d"].isna().sum())
    if n_novol:
        log(
            f"[targets] realized_vol_5d: {n_novol} rows without a full "
            f"{REALIZED_VOL_WINDOW}d forward window -> NaN"
        )
    return out[target_column_names(config)]
