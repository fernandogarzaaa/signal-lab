"""M4: event study. Do sentiment spikes predict abnormal returns?

Null hypothesis H0: the mean cumulative abnormal return (CAR) around
high-sentiment event days is zero; sentiment spikes carry no price
information beyond the market benchmark.

Method: for each event (ticker, event_date), compute daily abnormal returns
AR = R_ticker - R_market over trading-day windows [-1,+1] and [-1,+5]
relative to the event date, sum to CAR, then test the cross-event mean CAR
with a two-sided one-sample t-test. Report mean, t-statistic, p-value, and
the 95% confidence interval.

One way this analysis could lie: event windows overlap. A single market-wide
shock (or one company's earnings week) can generate several "events" whose
windows share the same price moves, violating the independence the t-test
assumes and understating the true standard error. We report the overlap rate
so the reader can judge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats

WINDOWS = [(1, 1), (1, 5)]  # (trading days before, trading days after)


def _daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.sort_values(["ticker", "date"]).copy()
    p["ret"] = p.groupby("ticker")["close"].pct_change()
    return p[["ticker", "date", "ret"]].dropna()


def _car_for_event(dates: list, rets: dict, mkt: dict, idx: int, pre: int, post: int) -> float | None:
    lo, hi = idx - pre, idx + post
    if lo < 0 or hi >= len(dates):
        return None
    car = 0.0
    for j in range(lo, hi + 1):
        d = dates[j]
        r = rets.get(d)
        m = mkt.get(d)
        if r is None or m is None or not np.isfinite(r) or not np.isfinite(m):
            return None
        car += r - m
    return car


def event_cars(events: pd.DataFrame, prices: pd.DataFrame,
               benchmark: str = "SPY", pre: int = 1, post: int = 1) -> pd.DataFrame:
    """Per-event cumulative abnormal return over [event-pre, event+post]
    trading days. Returns DataFrame[ticker, event_date, car, win_lo, win_hi]."""
    dr = _daily_returns(prices)
    by_ticker = {t: g.reset_index(drop=True) for t, g in dr[dr.ticker != benchmark].groupby("ticker")}
    mkt = dr[dr.ticker == benchmark].set_index("date")["ret"].to_dict()
    rows = []
    for _, ev in events.iterrows():
        t, ed = ev["ticker"], ev["event_date"]
        g = by_ticker.get(t)
        if g is None:
            continue
        dates = list(g["date"])
        pos = int(np.searchsorted(np.array(dates, dtype="datetime64[D]"),
                                  np.datetime64(ed, "D")))
        if pos >= len(dates):
            continue
        car = _car_for_event(dates, dict(zip(g["date"], g["ret"])), mkt, pos, pre, post)
        if car is not None and np.isfinite(car):
            rows.append({"ticker": t, "event_date": ed, "car": car,
                         "win_lo": dates[max(0, pos - pre)],
                         "win_hi": dates[min(len(dates) - 1, pos + post)]})
    return pd.DataFrame(rows)


def event_study(events: pd.DataFrame, prices: pd.DataFrame,
                benchmark: str = "SPY", windows: list[tuple[int, int]] = WINDOWS) -> pd.DataFrame:
    """events: DataFrame[ticker, event_date (datetime.date), sentiment].
    prices: DataFrame[ticker, date (datetime.date), close].
    Returns one row per window: n, mean_car, t_stat, p_value, ci95_lo, ci95_hi, overlap_rate.
    """
    rows = []
    for pre, post in windows:
        ec = event_cars(events, prices, benchmark=benchmark, pre=pre, post=post)
        cars = ec["car"].values if not ec.empty else np.array([])
        used_windows = list(zip(ec["ticker"], ec["win_lo"], ec["win_hi"])) if not ec.empty else []
        n = len(cars)
        mean = float(cars.mean()) if n else float("nan")
        if n < 2:
            rows.append({"window": f"[-{pre},+{post}]", "n": n,
                         "mean_car": round(mean, 5) if n else float("nan"),
                         "t_stat": np.nan, "p_value": np.nan, "ci95_lo": np.nan,
                         "ci95_hi": np.nan, "overlap_rate": np.nan})
            continue
        se = float(cars.std(ddof=1) / np.sqrt(n))
        t_stat = mean / se if se > 0 else 0.0
        p_value = float(2 * sstats.t.sf(abs(t_stat), df=n - 1))
        ci = sstats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
        # overlap: fraction of event-date pairs sharing any window day
        overlap = 0
        pairs = 0
        for i in range(len(used_windows)):
            for j in range(i + 1, len(used_windows)):
                if used_windows[i][0] != used_windows[j][0]:
                    continue
                pairs += 1
                if not (used_windows[i][2] < used_windows[j][1] or used_windows[j][2] < used_windows[i][1]):
                    overlap += 1
        rows.append({"window": f"[-{pre},+{post}]", "n": n, "mean_car": round(mean, 5),
                     "t_stat": round(t_stat, 3), "p_value": round(p_value, 4),
                     "ci95_lo": round(ci[0], 5), "ci95_hi": round(ci[1], 5),
                     "overlap_rate": round(overlap / pairs, 3) if pairs else 0.0})
    return pd.DataFrame(rows)
