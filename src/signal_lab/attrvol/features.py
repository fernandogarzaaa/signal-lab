"""ATTRVOL attribution features: frozen 5-feature attribution leg, point-in-time.

Feature spec (docs/ATTRVOL_PREREGISTRATION.md, frozen):
- Event rows: trailing top-decile |abn_ret| per ticker (|abn_ret(t0)|
  >= 90th percentile of |abn_ret| over the trailing 252 trading days
  strictly before t0; at least 60 valid observations).
- driver_H: Herfindahl index over the two-leg market/idiosyncratic
  decomposition of the event-day return (beta from trailing 252-day
  OLS vs SPY, reused from eventvol).
- idio_share: idiosyncratic share of the move.
- ret_x_H: signed event-day return interacted with concentration.
- analogue_vol_mean / analogue_vol_std: mean and sample std of
  realized_vol_5d over the k=10 nearest same-ticker historical
  analogues by z-scored 20-day price-path cosine similarity.

Point-in-time contract: every input uses closes on days <= t0 only.
Analogue candidates s satisfy s + 5 trading days <= t0 (position
based), so their realized_vol_5d labels are knowable at t0's close.
Verified by the truncation test: features for t0 <= cut are identical
whether computed on the full panel or on the panel truncated at cut.

Rows that cannot be featurized (fewer than 3 valid analogues, zero
denominator in the decomposition) get NaN attribution features and
are dropped by the caller with a logged count (never filled).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from signal_lab.eventvol import features as eventvol_feat
from signal_lab.harvix import features as harvix_feat
from signal_lab.vol.universe import MARKET_LEG

ATTR_COLUMNS: list[str] = [
    "driver_H",
    "idio_share",
    "ret_x_H",
    "analogue_vol_mean",
    "analogue_vol_std",
]

# Arm column sets. HARVIX_COLUMNS is the frozen 34-feature HAR+VIX
# set; the challenger adds exactly the 5 frozen attribution features.
BASELINE_COLUMNS: list[str] = list(harvix_feat.HARVIX_COLUMNS)
CHALLENGER_COLUMNS: list[str] = BASELINE_COLUMNS + ATTR_COLUMNS

# Trailing window (trading days) for the event quantile and beta.
_EVENT_WINDOW = 252
_EVENT_MIN_OBS = 60
_EVENT_QUANTILE = 0.90
# Analogue search: candidates in [p - _ANALOGUE_LOOKBACK, p - _LABEL_HORIZON].
_ANALOGUE_LOOKBACK = 500
_LABEL_HORIZON = 5
_ANALOGUE_K = 10
_ANALOGUE_MIN = 3
_PATH_LEN = 20


def _panel_matrices(prices: pd.DataFrame):
    """Union-calendar log-return matrix and SPY returns.

    Returns (cal, tickers, r, r_spy) where r is [n_days x n_tickers]
    of daily log returns (NaN-padded first row) and r_spy is [n_days].
    """
    px = prices.sort_values("date")
    tickers = [t for t in px["ticker"].unique()]
    if MARKET_LEG not in tickers:
        raise ValueError(f"[attrvol] market leg {MARKET_LEG} missing")
    cal = pd.DatetimeIndex(sorted(px["date"].unique()))
    mat = px.pivot_table(index="date", columns="ticker", values="adj_close",
                         aggfunc="first").reindex(cal)
    tickers = list(mat.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(mat.to_numpy(dtype=float))
    r = np.diff(logc, axis=0, prepend=np.full((1, logc.shape[1]), np.nan))
    r_spy = r[:, tickers.index(MARKET_LEG)]
    return cal, tickers, r, r_spy


def detect_events(prices: pd.DataFrame) -> pd.DataFrame:
    """Event table: (ticker, t0) rows whose |abn_ret| is top-decile trailing.

    abn_ret(t0) = r_i(t0) - r_SPY(t0). The 90th percentile is computed
    over the trailing 252 trading days STRICTLY before t0 (shifted
    rolling quantile; the event day never enters its own threshold).
    A row is an event iff |abn_ret(t0)| >= Q90 and at least 60 valid
    trailing observations exist.
    """
    cal, tickers, r, r_spy = _panel_matrices(prices)
    abn = r - r_spy[:, None]
    rows: list[dict] = []
    for j, t in enumerate(tickers):
        if t == MARKET_LEG:
            continue
        a = pd.Series(np.abs(abn[:, j]), index=cal)
        # Strictly-trailing quantile: rolling window then shift(1).
        q90 = a.rolling(_EVENT_WINDOW, min_periods=_EVENT_MIN_OBS).quantile(
            _EVENT_QUANTILE).shift(1)
        n_obs = a.rolling(_EVENT_WINDOW, min_periods=1).count().shift(1)
        is_event = (a >= q90) & (n_obs >= _EVENT_MIN_OBS) & np.isfinite(a)
        for d in cal[is_event.fillna(False).to_numpy()]:
            rows.append({"ticker": t, "t0": d,
                         "abn_ret": float(abn[cal.get_loc(d), j])})
    out = pd.DataFrame(rows, columns=["ticker", "t0", "abn_ret"])
    return out


def _zscored_paths(r_col: np.ndarray) -> np.ndarray:
    """Z-scored trailing 20-day return paths, NaN-padded to full length.

    Row p holds the z-scored path of the 20 returns ending at day p
    (r[p-19..p]); NaN where fewer than 20 finite returns exist or the
    path std is zero.
    """
    n = len(r_col)
    paths = np.full((n, _PATH_LEN), np.nan)
    if n < _PATH_LEN:
        return paths
    view = np.lib.stride_tricks.sliding_window_view(r_col, _PATH_LEN)
    # view[p] covers r[p..p+19]; row p+19 of paths gets the path ending at p+19.
    # All-NaN windows (e.g. pre-listing history) warn under nanmean/nanstd;
    # they are expected and correctly masked invalid below, so silence them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(view, axis=1)
        sd = np.nanstd(view, axis=1, ddof=1)
    ok = np.isfinite(mu) & np.isfinite(sd) & (sd > 0) & np.isfinite(view).all(axis=1)
    z = (view - mu[:, None]) / np.where(sd[:, None] > 0, sd[:, None], np.nan)
    paths[_PATH_LEN - 1:] = np.where(ok[:, None], z, np.nan)
    return paths


def _analogue_stats(paths: np.ndarray, labels: np.ndarray,
                    event_pos: np.ndarray) -> np.ndarray:
    """Analogue-implied vol stats per event position: [mean, std].

    Candidates for event at position p: q in [p-500, p-5] with finite
    path and finite label. Cosine similarity on z-scored paths; top
    k=10; fewer than 3 valid candidates -> NaN row.
    """
    n = len(paths)
    # NOTE: plain sum (not nansum): rows of `paths` are all-finite or
    # all-NaN by construction, so a NaN path yields a NaN norm and is
    # excluded by the finite check below. nansum would map it to 0.0,
    # which is finite and would wrongly admit it as a candidate.
    norms = np.sqrt(np.sum(paths ** 2, axis=1))
    out = np.full((len(event_pos), 2), np.nan)
    for ei, p in enumerate(event_pos):
        if not np.isfinite(norms[p]):
            continue
        lo = max(0, p - _ANALOGUE_LOOKBACK)
        hi = p - _LABEL_HORIZON
        if hi <= lo:
            continue
        cand = np.arange(lo, hi)
        valid = np.isfinite(norms[cand]) & np.isfinite(labels[cand])
        cand = cand[valid]
        if len(cand) < _ANALOGUE_MIN:
            continue
        sims = (paths[cand] @ paths[p]) / (norms[cand] * norms[p])
        sims = np.where(np.isfinite(sims), sims, -np.inf)
        top = cand[np.argsort(sims, kind="stable")[-_ANALOGUE_K:]]
        vols = labels[top]
        out[ei, 0] = float(np.mean(vols))
        out[ei, 1] = float(np.std(vols, ddof=1)) if len(vols) > 1 else 0.0
    return out


def decompose_event(beta: float, r_spy_t0: float,
                    r_i_t0: float) -> tuple[float, float, float]:
    """Two-leg market/idiosyncratic decomposition of an event-day return.

    Returns (driver_H, idio_share, ret_x_H). driver_H is the Herfindahl
    index over the absolute legs (in [0.5, 1.0]); NaN triple when the
    denominator is zero or non-finite.
    """
    m = beta * r_spy_t0
    e = r_i_t0 - m
    denom = abs(m) + abs(e)
    if denom == 0 or not np.isfinite(denom):
        return (np.nan, np.nan, np.nan)
    s_m = abs(m) / denom
    s_e = abs(e) / denom
    driver_h = s_m * s_m + s_e * s_e
    return (driver_h, s_e, r_i_t0 * driver_h)


def add_attribution_features(event_frame: pd.DataFrame,
                             full_frame: pd.DataFrame,
                             prices: pd.DataFrame) -> pd.DataFrame:
    """Merge the 5 frozen attribution features onto event rows, point-in-time.

    ``event_frame``: (ticker, t0) event rows (from detect_events,
    merged onto the feature frame). ``full_frame``: the all-ticker-day
    frame carrying the ``target`` column, used for analogue labels.
    ``prices``: the price panel. Every per-row computation uses closes
    on days <= t0 only. Rows that cannot be featurized get NaN and are
    left for the caller to drop with a logged count.
    """
    cal, tickers, r, r_spy = _panel_matrices(prices)
    cal_ord = np.array([d.toordinal() for d in cal], dtype=np.int64)
    t2j = {t: j for j, t in enumerate(tickers)}

    # Per-ticker label lookup from the full frame: ordinal -> target.
    ff = full_frame[["ticker", "t0", "target"]].copy()
    ff["ord"] = pd.to_datetime(ff["t0"]).dt.normalize().map(
        lambda d: d.toordinal())
    label_by_ticker: dict[str, dict[int, float]] = {}
    for t, grp in ff.groupby("ticker"):
        label_by_ticker[t] = dict(
            zip(grp["ord"].to_numpy(), grp["target"].to_numpy(dtype=float)))

    out = event_frame.reset_index(drop=True).copy()
    feat = np.full((len(out), len(ATTR_COLUMNS)), np.nan)
    n_beta_fallback = 0

    # Group event rows by ticker for vectorized per-ticker work.
    out["ord"] = pd.to_datetime(out["t0"]).dt.normalize().map(
        lambda d: d.toordinal())
    for t, grp in out.groupby("ticker"):
        j = t2j[t]
        r_col = r[:, j]
        paths = _zscored_paths(r_col)
        labels = np.full(len(cal), np.nan)
        lab = label_by_ticker.get(t, {})
        for q, d_ord in enumerate(cal_ord):
            if d_ord in lab:
                labels[q] = lab[d_ord]
        pos = np.searchsorted(cal_ord, grp["ord"].to_numpy())
        # Guard: every event t0 must be a union-calendar trading day.
        if not np.array_equal(cal_ord[pos], grp["ord"].to_numpy()):
            bad = grp.loc[cal_ord[pos] != grp["ord"].to_numpy(), "t0"].min()
            raise ValueError(
                f"[attrvol] t0 {bad} is not a price-panel trading day")
        ana = _analogue_stats(paths, labels, pos)
        for ei, (idx, p) in enumerate(zip(grp.index, pos)):
            beta, fell_back = eventvol_feat.trailing_beta(
                r_col, r_spy, int(p))
            n_beta_fallback += int(fell_back)
            driver_h, idio_share, ret_x_h = decompose_event(
                beta, float(r_spy[p]), float(r_col[p]))
            feat[idx, 0] = driver_h
            feat[idx, 1] = idio_share
            feat[idx, 2] = ret_x_h
            feat[idx, 3] = ana[ei, 0]
            feat[idx, 4] = ana[ei, 1]

    print(f"[attrvol] beta fallbacks: {n_beta_fallback} / {len(out)} "
          f"event rows", flush=True)
    for ci, c in enumerate(ATTR_COLUMNS):
        out[c] = feat[:, ci]
    return out.drop(columns=["ord"])
