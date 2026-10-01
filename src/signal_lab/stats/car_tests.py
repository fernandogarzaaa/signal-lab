"""Cross-sectional significance tests for cumulative abnormal returns.

The event study (``signal_lab.stats.event_study``) tested mean CAR with
a plain cross-sectional t-test only. That test is fragile in two known
ways:

1. Event-induced variance: if volatility rises around events, the
   cross-sectional CAR variance overstates the null variance and the
   plain t-test loses power / misstates size. The Boehmer-Musumeci-
   Poulsen (1991) BMP test standardizes each CAR by its own
   estimation-window volatility first.
2. Cross-sectional correlation: overlapping event windows (or a common
   market shock behind several events) correlate abnormal returns, which
   understates the true standard error. The Kolari-Pynnonen (2010)
   adjusted BMP scales the statistic by the mean pairwise
   estimation-window correlation.

This module adds, in our typed style (pattern from
davidkreitmeir/clascestudy, reimplemented, not copied):

- ``event_ar_panel``: per-event daily abnormal returns over an
  estimation window and the event window, from the same market-adjusted
  model as ``event_cars`` (AR = R_ticker - R_market).
- ``cross_sectional_tests``: for one event window, the plain t-test on
  raw CARs plus three standardized tests on SCARs:
    * BMP:  t = mean(SCAR) / (sd(SCAR)/sqrt(N))
    * adj-BMP: BMP scaled by sqrt((1 - r_bar)/(1 + (N-1) r_bar)) with
      r_bar = mean pairwise correlation of estimation-window ARs
    * GRANK: Kolari-Pynnonen-style generalized rank test. Each event's
      SCAR is ranked among its own estimation-window rolling L-day
      SCARs (standardized the same way); the standardized ranks are
      tested against their null expectation. Robust to non-normality
      and event-induced variance.

Standardization: SCAR_i = CAR_i / (s_i * sqrt(L)), s_i = estimation
standard deviation of daily AR (ddof=1), L = event-window length in
trading days. Events with < 30 estimation days or zero estimation
variance are dropped (reported in ``n_dropped``), never zero-filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats

MIN_EST_DAYS = 30
MIN_EVENTS = 3
# Estimation windows with std below this are degenerate (a constant price
# series has std ~1e-19 in floating point, not exactly 0.0); their SCARs
# would be astronomical garbage, so the events are dropped.
MIN_EST_STD = 1e-10

TEST_NAMES = ("t", "bmp", "adj_bmp", "grank")


def event_ar_panel(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    pre: int = 1,
    post: int = 1,
    est_days: int = 120,
) -> pd.DataFrame:
    """Per-event daily abnormal returns.

    For each event (ticker, event_date): the ``est_days`` trading days
    ending ``pre + 1`` days before the event (estimation window, strictly
    before the event window) plus the event window
    ``[event - pre, event + post]``.

    Returns a long DataFrame with columns
    [event_id, ticker, event_date, offset, ar, in_event], where offset is
    the trading-day offset from the event date and in_event marks the
    event window. Events missing any required day are dropped.
    """
    from signal_lab.stats import _daily_returns

    dr = _daily_returns(prices)
    by_ticker = {
        t: g.reset_index(drop=True) for t, g in dr[dr.ticker != benchmark].groupby("ticker")
    }
    mkt = dr[dr.ticker == benchmark].set_index("date")["ret"].to_dict()

    rows = []
    for eid, ev in enumerate(events.itertuples()):
        t, ed = ev.ticker, ev.event_date
        g = by_ticker.get(t)
        if g is None:
            continue
        dates = list(g["date"])
        pos = int(
            np.searchsorted(np.array(dates, dtype="datetime64[D]"), np.datetime64(ed, "D"))
        )
        if pos >= len(dates):
            continue
        lo, hi = pos - pre - est_days, pos + post
        if lo < 0 or hi >= len(dates):
            continue
        rets = dict(zip(g["date"], g["ret"]))
        block = []
        ok = True
        for j in range(lo, hi + 1):
            d = dates[j]
            r, m = rets.get(d), mkt.get(d)
            if r is None or m is None or not np.isfinite(r) or not np.isfinite(m):
                ok = False
                break
            block.append((j - pos, r - m))
        if not ok:
            continue
        for offset, ar in block:
            rows.append(
                {
                    "event_id": eid,
                    "ticker": t,
                    "event_date": ed,
                    "offset": offset,
                    "ar": ar,
                    "in_event": bool(-pre <= offset <= post),
                }
            )
    cols = ["event_id", "ticker", "event_date", "offset", "ar", "in_event"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def _scar_frame(panel: pd.DataFrame, pre: int, post: int) -> pd.DataFrame:
    """One row per event: car, s_est, scar, plus the estimation AR series."""
    L = pre + post + 1
    out = []
    for eid, g in panel.groupby("event_id"):
        est = g[~g["in_event"]].sort_values("offset")["ar"].to_numpy()
        evt = g[g["in_event"]].sort_values("offset")["ar"].to_numpy()
        if len(est) < MIN_EST_DAYS or len(evt) != L:
            continue
        s = float(np.std(est, ddof=1))
        if not np.isfinite(s) or s < MIN_EST_STD:
            continue
        car = float(evt.sum())
        out.append(
            {
                "event_id": eid,
                "ticker": g["ticker"].iloc[0],
                "car": car,
                "s_est": s,
                "scar": car / (s * np.sqrt(L)),
                "est_ars": est,
            }
        )
    return pd.DataFrame(out)


def _mean_pairwise_corr(est_series: list[np.ndarray]) -> float:
    """Mean pairwise Pearson correlation of estimation AR series.

    Series are aligned on their trailing (most recent) days; pairs with
    fewer than 30 overlapping days are skipped. Returns 0.0 when no
    usable pair exists (then adj-BMP reduces to BMP).
    """
    corrs = []
    for i in range(len(est_series)):
        for j in range(i + 1, len(est_series)):
            a, b = est_series[i], est_series[j]
            n = min(len(a), len(b))
            if n < MIN_EST_DAYS:
                continue
            aa, bb = a[-n:], b[-n:]
            if np.std(aa, ddof=1) < MIN_EST_STD or np.std(bb, ddof=1) < MIN_EST_STD:
                continue
            c = float(np.corrcoef(aa, bb)[0, 1])
            if np.isfinite(c):
                corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0


def _t_pvalue(t_stat: float, n: int) -> float:
    return float(2 * sstats.t.sf(abs(t_stat), df=n - 1))


def bmp_test(scars: np.ndarray) -> tuple[float, float]:
    """Boehmer-Musumeci-Poulsen (1991) standardized cross-sectional test."""
    n = len(scars)
    sd = float(np.std(scars, ddof=1))
    t = float(scars.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0
    return t, _t_pvalue(t, n)


def adj_bmp_test(scars: np.ndarray, r_bar: float) -> tuple[float, float, float]:
    """Kolari-Pynnonen (2010) correlation-adjusted BMP.

    Returns (t_stat, p_value, r_bar_used). With r_bar = 0 this is BMP.
    """
    n = len(scars)
    t_bmp, _ = bmp_test(scars)
    adj = np.sqrt((1.0 - r_bar) / (1.0 + (n - 1) * r_bar))
    t_adj = float(t_bmp * adj)
    return t_adj, _t_pvalue(t_adj, n), float(r_bar)


def grank_test(scars: np.ndarray, est_series: list[np.ndarray], L: int) -> tuple[float, float]:
    """Generalized rank test on standardized CARs (Kolari-Pynnonen style).

    For each event, the SCAR is ranked among its own estimation-window
    rolling L-day SCARs (same standardization). Under H0 the rank is
    uniform on {1..K}; standardized rank deviations are tested with a
    cross-sectional t. Robust to non-normal ARs and event-induced
    variance by construction.
    """
    z = []
    for scar, est in zip(scars, est_series):
        s = float(np.std(est, ddof=1))
        if not np.isfinite(s) or s < MIN_EST_STD or len(est) < L:
            continue
        roll = np.array(
            [est[j:j + L].sum() / (s * np.sqrt(L)) for j in range(len(est) - L + 1)]
        )
        pool = np.append(roll, scar)
        # Average rank of the event SCAR within the pooled values
        # (rank 1 = smallest). Ties get the average rank.
        rank = float(sstats.rankdata(pool)[-1])
        K = len(pool)
        expected = (K + 1) / 2.0
        var = (K * K - 1) / 12.0
        z.append((rank - expected) / np.sqrt(var))
    z = np.asarray(z)
    n = len(z)
    if n < MIN_EVENTS:
        return float("nan"), float("nan")
    sd = float(np.std(z, ddof=1))
    t = float(z.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0
    return t, _t_pvalue(t, n)


def cross_sectional_tests(
    panel: pd.DataFrame, pre: int, post: int
) -> dict:
    """All four CAR significance tests for one event window.

    Returns a dict with per-test (stat, p_value, n) plus diagnostics:
    n_events, n_dropped, r_bar (mean estimation-window correlation).
    Tests with fewer than MIN_EVENTS usable events return NaN stats
    rather than a number that pretends to mean something.
    """
    L = pre + post + 1
    sf = _scar_frame(panel, pre, post)
    n_dropped = int(panel["event_id"].nunique() - len(sf)) if not panel.empty else 0
    result: dict = {
        "n_events": int(len(sf)),
        "n_dropped": n_dropped,
        "window": f"[-{pre},+{post}]",
    }
    if len(sf) < MIN_EVENTS:
        for name in TEST_NAMES:
            result[name] = {"stat": float("nan"), "p_value": float("nan"),
                            "n": int(len(sf))}
        result["r_bar"] = float("nan")
        return result

    cars = sf["car"].to_numpy()
    scars = sf["scar"].to_numpy()
    est_series = list(sf["est_ars"])

    # Plain cross-sectional t on raw CARs (matches event_study's test).
    sd = float(np.std(cars, ddof=1))
    t_plain = float(cars.mean() / (sd / np.sqrt(len(cars)))) if sd > 0 else 0.0
    result["t"] = {"stat": round(t_plain, 3), "p_value": round(_t_pvalue(t_plain, len(cars)), 4),
                   "n": int(len(cars))}

    t_bmp, p_bmp = bmp_test(scars)
    result["bmp"] = {"stat": round(t_bmp, 3), "p_value": round(p_bmp, 4),
                     "n": int(len(scars))}

    r_bar = _mean_pairwise_corr(est_series)
    t_adj, p_adj, _ = adj_bmp_test(scars, r_bar)
    result["adj_bmp"] = {"stat": round(t_adj, 3), "p_value": round(p_adj, 4),
                         "n": int(len(scars))}
    result["r_bar"] = round(r_bar, 4)

    t_gr, p_gr = grank_test(scars, est_series, L)
    result["grank"] = {"stat": round(t_gr, 3) if np.isfinite(t_gr) else float("nan"),
                       "p_value": round(p_gr, 4) if np.isfinite(p_gr) else float("nan"),
                       "n": int(len(scars))}
    return result
