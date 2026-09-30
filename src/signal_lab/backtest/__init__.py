"""M5: walk-forward backtest. Compare strategies without fooling yourself.

Core engine: run(signals, prices, cost_bps). `signals` maps a strategy name
to a signal frame with columns (ticker, asof, strength):

  asof     when the signal became known / takes effect
  strength position size in [-1, 1] held from asof until superseded

No-lookahead mechanism: positions for date d are joined with merge_asof on
(asof <= d) per ticker, so a signal can never influence a position dated
before it was known. tests/test_backtest.py proves this behaviorally: a
signal stamped asof=2026-01-06 cannot capture the price jump of 2026-01-05
to 2026-01-06, because the position held over that jump was decided from
signals known on 2026-01-05. Honest asof stamping is the caller's
responsibility; the engine's contract is mechanical: position(d) depends
only on signals with asof <= d.

Strategies (built by run_backtest.py):
  sentiment_momentum  long tickers for `hold_days` after a sentiment spike
  buy_and_hold        equal-weight long every ticker, every day
  random_entry        same number of entries as sentiment_momentum, but on
                      random ticker-days (seeded): the "is this skill or luck"
                      control

Costs: `cost_bps` per one-way trade on turnover = |pos_d - pos_{d-1}| / 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats

TRADING_DAYS = 252


def _panel(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.sort_values(["ticker", "date"]).copy()
    p["ret_next"] = p.groupby("ticker")["close"].pct_change().shift(-1)
    return p[["ticker", "date", "ret_next"]].dropna().reset_index(drop=True)


def _positions(signals: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Join signals onto the price panel with merge_asof on asof <= date.

    Each signal row (ticker, asof, strength) takes effect from asof until a
    later signal row for the same ticker supersedes it.
    """
    sig = signals.copy()
    sig["asof"] = pd.to_datetime(sig["asof"])
    # later rows win on ties; max strength wins on identical asof
    sig = sig.sort_values(["ticker", "asof", "strength"]).drop_duplicates(
        subset=["ticker", "asof"], keep="last")
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    out_frames = []
    for t, g in panel.groupby("ticker"):
        s = sig[sig["ticker"] == t].sort_values("asof")
        g = g.sort_values("date")
        if s.empty:
            g["strength"] = 0.0
        else:
            m = pd.merge_asof(g, s[["asof", "strength"]], left_on="date",
                              right_on="asof", direction="backward")
            m["strength"] = m["strength"].fillna(0.0)
            g = m
        out_frames.append(g)
    out = pd.concat(out_frames, ignore_index=True).sort_values(["date", "ticker"])
    return out


def _daily_pnl(pos: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    pos = pos.sort_values(["ticker", "date"]).copy()
    pos["prev_strength"] = pos.groupby("ticker")["strength"].shift(1).fillna(0.0)
    pos["turnover"] = (pos["strength"] - pos["prev_strength"]).abs() / 2.0
    pos["cost"] = pos["turnover"] * cost_bps / 1e4
    pos["pnl"] = pos["strength"] * pos["ret_next"] - pos["cost"]
    daily = pos.groupby("date").agg(port_ret=("pnl", "mean")).reset_index()
    return daily


def _metrics(daily: pd.DataFrame) -> dict:
    r = daily["port_ret"].values
    n = len(r)
    if n < 2:
        return {"n_days": n, "total_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    cum = np.cumprod(1 + r)
    total = float(cum[-1] - 1)
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else 0.0
    dd = float((np.maximum.accumulate(cum) - cum).max() / np.maximum.accumulate(cum).max())
    return {"n_days": n, "total_return": round(total, 4), "sharpe": round(sharpe, 3),
            "max_drawdown": round(dd, 4)}


def bootstrap_sharpe_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 5000,
                          seed: int = 7) -> tuple[float, float]:
    """95% CI for Sharpe(a) - Sharpe(b) via resampling days with replacement."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        sa = a[idx].mean() / a[idx].std(ddof=1) * np.sqrt(TRADING_DAYS) if a[idx].std(ddof=1) > 0 else 0.0
        sb = b[idx].mean() / b[idx].std(ddof=1) * np.sqrt(TRADING_DAYS) if b[idx].std(ddof=1) > 0 else 0.0
        diffs.append(sa - sb)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run(signals: dict[str, pd.DataFrame], prices: pd.DataFrame,
        cost_bps: float = 5.0) -> pd.DataFrame:
    """Run every strategy in `signals` over `prices`.

    Returns one row per strategy: n_days, total_return, sharpe, max_drawdown,
    plus paired t-test and bootstrap CI of (strategy - buy_and_hold) daily
    returns when a 'buy_and_hold' strategy is present.
    """
    panel = _panel(prices)
    daily_rets: dict[str, pd.DataFrame] = {}
    rows = []
    for name, sig in signals.items():
        pos = _positions(sig, panel)
        daily = _daily_pnl(pos, cost_bps)
        daily_rets[name] = daily
        rows.append({"strategy": name, **_metrics(daily)})
    res = pd.DataFrame(rows)
    if "buy_and_hold" in daily_rets:
        base = daily_rets["buy_and_hold"].set_index("date")["port_ret"]
        for name, daily in daily_rets.items():
            if name == "buy_and_hold":
                continue
            d = daily.set_index("date")["port_ret"]
            aligned = pd.concat([d, base], axis=1, join="inner").dropna()
            # Coerce: an object-dtype port_ret (e.g. from concatenating an
            # empty signal frame) crashes scipy's ttest_rel. Non-numeric
            # entries become NaN and are dropped before the test.
            a = pd.to_numeric(aligned.iloc[:, 0], errors="coerce")
            b = pd.to_numeric(aligned.iloc[:, 1], errors="coerce")
            mask = a.notna() & b.notna()
            a, b = a[mask], b[mask]
            if len(a) < 3:
                continue
            t_stat, p_val = sstats.ttest_rel(a, b)
            lo, hi = bootstrap_sharpe_diff(a.values, b.values)
            res.loc[res.strategy == name, "paired_t"] = round(float(t_stat), 3)
            res.loc[res.strategy == name, "paired_p"] = round(float(p_val), 4)
            res.loc[res.strategy == name, "sharpe_diff_ci95"] = f"[{lo:.2f}, {hi:.2f}]"
    return res
