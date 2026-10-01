"""Market-context features for the stage-3 classifier (0.3.0 workstream 3).

Seven dense features, every one knowable at prediction time. All price
inputs come from trading days strictly BEFORE the article's publish date,
so no future information can leak into the model. Metadata features come
from the article row itself (URL, timestamp, text).

Price features (per ticker, as of the last trading day < pub_date):

- ``ctx_vol20_pct``: percentile of the trailing-20-trading-day realized
  volatility (sample std, ddof=1, of daily close-to-close returns) within
  its own trailing history of up to 252 trading days (minimum 20
  observations). 1.0 = the most volatile the ticker has been in the
  lookback window.
- ``ctx_mom5``: trailing 5-trading-day return, ``close[t-1]/close[t-6]-1``
  where ``t`` is the as-of trading day (equivalently the product of the
  last 5 daily returns minus 1).
- ``ctx_mom20``: trailing 20-trading-day return, same construction.
- ``ctx_sector_rel5``: ticker trailing-5d return minus the sector leg's
  trailing-5d return over the same window.

Article metadata (known at publish time):

- ``ctx_src_tier``: source-domain tier, 2 = wire service / top financial
  outlet, 1 = established press, 0 = other or unknown. The mapping is a
  heuristic over outlet reputation for financial news, documented below.
- ``ctx_word_count``: word count of title + body snippet.
- ``ctx_hour_bucket``: UTC publish hour bucketed into quarters of the day:
  0 = 00-05, 1 = 06-11, 2 = 12-17, 3 = 18-23.

Sector mapping: a static ticker -> (GICS sector, sector ETF) table. The
assignments follow the standard GICS classification. Sector ETF prices are
NOT part of the price backfill, so the sector leg falls back to the market
benchmark (SPY) return; ``build_context_features`` accepts an optional
``sector_prices`` frame so a future backfill can swap the real ETF leg in
without changing callers. While the fallback is active, ``ctx_sector_rel5``
equals the ticker's 5-day abnormal return vs the benchmark.

The no-leakage rule is structural: the as-of trading day for an article
published on calendar date D is the latest trading day strictly before D
(``searchsorted(..., side="left") - 1``), and every rolling window ends on
or before that day. ``tests/test_context_features.py`` pins this with a
synthetic post-article price spike that must leave features unchanged.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

CONTEXT_FEATURE_NAMES = [
    "ctx_vol20_pct",
    "ctx_mom5",
    "ctx_mom20",
    "ctx_sector_rel5",
    "ctx_src_tier",
    "ctx_word_count",
    "ctx_hour_bucket",
]

# Static ticker -> (GICS sector, sector ETF). Assignments follow the standard
# GICS classification; the ETF is the corresponding Select Sector SPDR.
TICKER_SECTORS: dict[str, dict[str, str]] = {
    "AAPL": {"sector": "Information Technology", "etf": "XLK"},
    "MSFT": {"sector": "Information Technology", "etf": "XLK"},
    "NVDA": {"sector": "Information Technology", "etf": "XLK"},
    "GOOGL": {"sector": "Communication Services", "etf": "XLC"},
    "META": {"sector": "Communication Services", "etf": "XLC"},
    "AMZN": {"sector": "Consumer Discretionary", "etf": "XLY"},
    "TSLA": {"sector": "Consumer Discretionary", "etf": "XLY"},
    "JPM": {"sector": "Financials", "etf": "XLF"},
    "JNJ": {"sector": "Health Care", "etf": "XLV"},
    "XOM": {"sector": "Energy", "etf": "XLE"},
}

# Source-domain tiers. Heuristic over outlet reputation for financial news:
# tier 2 = wire services and top financial outlets, tier 1 = established
# general/regional press and large finance portals, tier 0 = everything else
# (unknown domains, blogs, aggregators without a newsroom). Matching is by
# domain suffix, so subdomains (e.g. markets.businessinsider.com) inherit
# their parent's tier.
_TIER2_DOMAINS = frozenset(
    {
        "reuters.com",
        "bloomberg.com",
        "wsj.com",
        "ft.com",
        "cnbc.com",
        "apnews.com",
        "marketwatch.com",
        "barrons.com",
        "economist.com",
        "nytimes.com",
        "washingtonpost.com",
        "investors.com",
        "fortune.com",
        "forbes.com",
        "morningstar.com",
    }
)
_TIER1_DOMAINS = frozenset(
    {
        "yahoo.com",
        "nasdaq.com",
        "zacks.com",
        "fool.com",
        "benzinga.com",
        "investing.com",
        "marketbeat.com",
        "gurufocus.com",
        "thestreet.com",
        "seekingalpha.com",
        "businessinsider.com",
        "usatoday.com",
        "latimes.com",
        "chicagotribune.com",
        "nypost.com",
        "theguardian.com",
        "bbc.com",
        "bbc.co.uk",
        "abcnews.go.com",
        "cbsnews.com",
        "nbcnews.com",
        "cnn.com",
        "foxbusiness.com",
        "foxnews.com",
        "sandiegouniontribune.com",
        "denverpost.com",
    }
)

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# Rolling lookbacks, in trading days. Kept as constants so the leakage audit
# can cite them by name.
MOM_SHORT = 5
MOM_LONG = 20
VOL_WINDOW = 20
VOL_HISTORY = 252
VOL_HISTORY_MIN = 20


def source_domain_tier(url: str) -> int:
    """2/1/0 tier for an article URL by domain suffix (0 when unknown)."""
    try:
        host = urlparse(url or "").netloc.lower()
    except Exception:  # noqa: BLE001 - unparseable URL is tier 0, not an error
        return 0
    host = host.removeprefix("www.")
    if not host:
        return 0
    parts = host.split(".")
    for i in range(len(parts)):
        suffix = ".".join(parts[i:])
        if suffix in _TIER2_DOMAINS:
            return 2
        if suffix in _TIER1_DOMAINS:
            return 1
    return 0


def _trailing_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, trading date): trailing momentum/volatility features.

    Each row's features use only daily returns ending on that row's date
    (never later): mom5/mom20 need 5/20 full returns, vol20 needs 20, and
    vol20_pct ranks the current 20d vol against its own trailing history.
    """
    p = prices[["ticker", "date", "close"]].copy()
    p["date"] = pd.to_datetime(p["date"]).dt.date
    p = p.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])
    p = p.reset_index(drop=True)
    p["ret"] = p.groupby("ticker")["close"].pct_change()

    ret_by_ticker = p["ret"].groupby(p["ticker"])
    p["mom5"] = ret_by_ticker.transform(
        lambda s: (
            (1.0 + s).rolling(MOM_SHORT, min_periods=MOM_SHORT).apply(np.prod, raw=True)
            - 1.0
        )
    )
    p["mom20"] = ret_by_ticker.transform(
        lambda s: (
            (1.0 + s).rolling(MOM_LONG, min_periods=MOM_LONG).apply(np.prod, raw=True)
            - 1.0
        )
    )
    p["vol20"] = ret_by_ticker.transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
    )

    def _percentile_of_current(window: np.ndarray) -> float:
        # Fraction of the valid trailing history at or below the current
        # value. NaN history (the first VOL_WINDOW-1 trading days, where no
        # 20d vol exists yet) is excluded, and fewer than VOL_HISTORY_MIN
        # valid observations yields NaN rather than a noisy rank.
        w = window[~np.isnan(window)]
        if len(w) < VOL_HISTORY_MIN + 1:
            return np.nan
        return float((w[:-1] <= w[-1]).mean())

    p["vol20_pct"] = (
        p.groupby("ticker")["vol20"]
        .transform(
            lambda s: s.rolling(VOL_HISTORY, min_periods=VOL_HISTORY_MIN).apply(
                _percentile_of_current, raw=True
            )
        )
        .astype(float)
    )
    return p[["ticker", "date", "mom5", "mom20", "vol20_pct"]]


def _ordinal(d) -> int:
    return pd.Timestamp(d).date().toordinal()


def _asof_positions(
    calendars: dict[str, np.ndarray], tickers: np.ndarray, dates
) -> np.ndarray:
    """For each article: position of the latest trading day strictly before
    its publish date in its ticker's calendar (-1 when none exists)."""
    pub_ord = np.array([_ordinal(d) for d in dates])
    pos = np.full(len(tickers), -1, dtype=int)
    for ticker, cal in calendars.items():
        m = tickers == ticker
        if m.any():
            pos[m] = np.searchsorted(cal, pub_ord[m], side="left") - 1
    return pos


def build_context_features(
    news: pd.DataFrame,
    prices: pd.DataFrame,
    sector_prices: pd.DataFrame | None = None,
    benchmark: str = "SPY",
    log=print,
) -> pd.DataFrame:
    """Compute the 7 market-context features for each news row.

    ``news`` needs columns: url, ticker, published_at, pub_date, text.
    ``prices`` needs columns: ticker, date, close. ``sector_prices``
    (optional) needs the same columns for sector ETFs; when omitted, the
    sector leg falls back to ``benchmark`` and the choice is logged.

    Returns a DataFrame with columns ``["url", "ctx_asof"] +
    CONTEXT_FEATURE_NAMES``, in the input row order and on the input
    index. ``ctx_asof`` is the latest trading day strictly before the
    publish date (the point-in-time anchor; None where none exists).
    Price features are NaN for rows whose ticker has insufficient
    pre-publication history; the caller decides how to handle them
    (the stage-3 pipeline drops and counts them).
    """
    news = news.copy()
    if "pub_date" not in news.columns:
        news["pub_date"] = pd.to_datetime(news["published_at"], utc=True).dt.date
    feats = _trailing_price_features(prices)
    calendars = {
        t: np.array([_ordinal(d) for d in sub["date"]])
        for t, sub in feats.groupby("ticker")
    }

    tickers = news["ticker"].to_numpy()
    pos = _asof_positions(calendars, tickers, news["pub_date"].to_numpy())

    out = pd.DataFrame(np.nan, index=news.index, columns=CONTEXT_FEATURE_NAMES)
    ok = pos >= 0
    if ok.any():
        for ticker, sub in feats.groupby("ticker"):
            m = ok & (tickers == ticker)
            if not m.any():
                continue
            rows = (
                sub.sort_values("date")
                .reset_index(drop=True)[["mom5", "mom20", "vol20_pct"]]
                .to_numpy()
            )
            vals = rows[pos[m]]
            out.loc[m, ["ctx_mom5", "ctx_mom20", "ctx_vol20_pct"]] = vals[:, [0, 1, 2]]

    # Sector-relative strength: ticker mom5 minus the sector leg's mom5 at
    # the same as-of date. The leg uses real ETF prices when provided,
    # otherwise the market benchmark (logged below).
    if sector_prices is not None:
        leg_feats = _trailing_price_features(sector_prices)
        leg_for = {t: info["etf"] for t, info in TICKER_SECTORS.items()}
        leg_source = "sector ETF prices"
    else:
        leg_feats = feats[feats["ticker"] == benchmark].copy()
        leg_for = {}
        leg_source = f"{benchmark} (fallback: no sector ETF prices in the backfill)"
    log(f"[ws3] sector leg for ctx_sector_rel5: {leg_source}")
    leg_cals = {
        t: np.array([_ordinal(d) for d in sub["date"]])
        for t, sub in leg_feats.groupby("ticker")
    }
    leg_mom5 = {
        t: sub.sort_values("date").reset_index(drop=True)["mom5"].to_numpy()
        for t, sub in leg_feats.groupby("ticker")
    }
    # As-of ordinals per article (from the ticker calendars above).
    asof_ord = np.full(len(news), -1, dtype=int)
    for ticker, cal in calendars.items():
        m = ok & (tickers == ticker)
        asof_ord[m] = cal[pos[m]]
    for ticker in np.unique(tickers[ok]):
        m = ok & (tickers == ticker)
        leg_ticker = leg_for.get(ticker, benchmark)
        cal = leg_cals.get(leg_ticker)
        mom5 = leg_mom5.get(leg_ticker)
        if cal is None or mom5 is None:
            continue
        leg_pos = np.searchsorted(cal, asof_ord[m], side="right") - 1
        leg_ok = leg_pos >= 0
        idx = np.where(m)[0][leg_ok]
        out.loc[out.index[idx], "ctx_sector_rel5"] = (
            out.loc[out.index[idx], "ctx_mom5"].to_numpy() - mom5[leg_pos[leg_ok]]
        )

    # A missing benchmark leg is a missing feature value, not a missing
    # observation: neutralize to 0.0 (no measured sector-relative signal)
    # instead of dropping the article. Dropping would silently discard
    # most of the backfill whenever the benchmark's price history is
    # shorter than the tickers' (e.g. SPY added late to the backfill).
    n_neutral = int(out["ctx_sector_rel5"].isna().sum())
    out["ctx_sector_rel5"] = out["ctx_sector_rel5"].fillna(0.0)
    if n_neutral:
        log(f"[ws3] ctx_sector_rel5 neutralized (0.0) for {n_neutral} rows: no benchmark leg")
    out["ctx_src_tier"] = [source_domain_tier(u) for u in news["url"]]
    out["ctx_word_count"] = (
        news["text"].fillna("").astype(str).apply(lambda s: len(_WORD_RE.findall(s)))
    ).astype(float)
    hours = pd.to_datetime(news["published_at"], utc=True).dt.hour
    out["ctx_hour_bucket"] = (hours // 6).clip(0, 3).astype(float)

    result = pd.DataFrame({"url": news["url"].to_numpy()}, index=news.index)
    # ctx_asof: the point-in-time anchor, the latest trading day strictly
    # before the publish date (None where no prior trading day exists).
    # Used by validation.point_in_time to prove features predate publication.
    asof_dates = np.full(len(news), None, dtype=object)
    for ticker, cal in calendars.items():
        m = (tickers == ticker) & (pos >= 0)
        if m.any():
            cal_dates = np.array(
                [pd.Timestamp.fromordinal(int(o)).date() for o in cal]
            )
            asof_dates[m] = cal_dates[pos[m]]
    result["ctx_asof"] = asof_dates
    return pd.concat([result, out], axis=1)


# ---------------------------------------------------------------------------
# Phase 2: trailing price/volume features (research feature set).
#
# Fourteen dense features, every one knowable at prediction time. All price
# inputs come from trading days on or before the as-of day D (the latest
# trading day strictly BEFORE the article's publish date -- the same rule
# as the ctx_* features, via the shared _asof_positions helper), so no
# future information can leak into the model. Insufficient history yields
# NaN, never filled, never forward-filled.
#
# - ``px_ret_1d``: trailing 1-day return, close[D]/close[D-1] - 1.
#   (ctx_mom5/ctx_mom20 already cover the 5d/20d trailing returns, so the
#   1d return is the non-duplicating addition.)
# - ``px_rsi_14``: Wilder's RSI(14), 0-100. Seeded with the simple mean of
#   the first 14 gains/losses, then Wilder smoothing
#   avg_t = (avg_{t-1}*13 + x_t)/14. First value at the 15th close.
# - ``px_macd_hist``: MACD(12,26,9) histogram = MACD line - signal line,
#   with the recursive EMA (ema[0] = x[0]). NaN until 35 closes are
#   available (26 for the slow EMA to converge + 9 for the signal line).
# - ``px_ma_dist_20`` / ``px_ma_dist_50``: (close[D] - SMA_n[D]) / SMA_n[D].
# - ``px_realvol_5d`` / ``px_realvol_20d``: sample std (ddof=1) of daily
#   LOG returns over the trailing 5/20 trading days ending on D, in daily
#   units (not annualized).
# - ``px_atr_14``: Wilder's ATR(14) divided by close[D] (fraction of price).
#   True range = max(high-low, |high-prev_close|, |low-prev_close|),
#   seeded with the mean of the first 14 true ranges.
# - ``px_vol_z_20``: (volume[D] - mean(volume[D-19..D])) / std(...),
#   ddof=1; exactly 0.0 when the trailing volume is constant.
# - ``px_relvol_20``: volume[D] / median(volume[D-19..D]).
# - ``px_vs_spy_5d`` / ``px_vs_spy_20d``: ticker trailing return minus the
#   SPY trailing return over the same window ending on D.
# - ``mkt_spy_ret_20d``: SPY trailing 20d return ending on D.
# - ``mkt_spy_vol_20d``: sample std (ddof=1) of SPY daily log returns over
#   the trailing 20 trading days ending on D, daily units.
#
# VIX: deliberately skipped. There is no local VIX series in the DuckDB
# backfill, and adding one would put a network dependency in the research
# path. mkt_spy_ret_20d / mkt_spy_vol_20d cover the market-regime role the
# VIX leg would play.
# ---------------------------------------------------------------------------

PRICE_FEATURE_NAMES = [
    "px_ret_1d",
    "px_rsi_14",
    "px_macd_hist",
    "px_ma_dist_20",
    "px_ma_dist_50",
    "px_realvol_5d",
    "px_realvol_20d",
    "px_atr_14",
    "px_vol_z_20",
    "px_relvol_20",
    "px_vs_spy_5d",
    "px_vs_spy_20d",
    "mkt_spy_ret_20d",
    "mkt_spy_vol_20d",
]

# Warmup conventions, in trading days. A feature is NaN until its full
# warmup is available; warmups are never filled or forward-filled.
RSI_N = 14  # Wilder RSI: 14 price changes to seed
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MACD_MIN_BARS = 35  # 26 closes for the slow EMA to converge + 9 for signal
ATR_N = 14  # Wilder ATR: 14 true ranges to seed


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """Recursive EMA (the adjust=False convention).

    ema[0] = x[0]; ema[t] = alpha*x[t] + (1-alpha)*ema[t-1] with
    alpha = 2/(span+1). Defined from the first bar, so callers impose
    their own warmup cutoff (see MACD_MIN_BARS).
    """
    values = np.asarray(values, dtype=float)
    out = np.empty(len(values), dtype=float)
    if len(values) == 0:
        return out
    alpha = 2.0 / (span + 1)
    out[0] = values[0]
    for t in range(1, len(values)):
        out[t] = alpha * values[t] + (1.0 - alpha) * out[t - 1]
    return out


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _wilder_rsi(close: np.ndarray, n: int = RSI_N) -> np.ndarray:
    """Wilder's RSI: seed with the simple mean of the first n gains/losses,
    then Wilder smoothing avg_t = (avg_{t-1}*(n-1) + x_t)/n.

    The first value sits at bar n (it consumes n price changes); earlier
    bars are NaN.
    """
    close = np.asarray(close, dtype=float)
    out = np.full(len(close), np.nan)
    if len(close) <= n:
        return out
    delta = np.diff(close)
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)
    ag = float(gain[:n].mean())
    al = float(loss[:n].mean())
    out[n] = _rsi_from_avgs(ag, al)
    for i in range(n + 1, len(close)):
        ag = (ag * (n - 1) + gain[i - 1]) / n
        al = (al * (n - 1) + loss[i - 1]) / n
        out[i] = _rsi_from_avgs(ag, al)
    return out


def _wilder_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = ATR_N
) -> np.ndarray:
    """Wilder's ATR: true range seeded with the mean of the first n true
    ranges, then Wilder smoothing. First value at bar n-1; earlier NaN."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    out = np.full(len(close), np.nan)
    if len(close) < n:
        return out
    tr = np.empty(len(close), dtype=float)
    tr[0] = high[0] - low[0]
    for j in range(1, len(close)):
        tr[j] = max(
            high[j] - low[j],
            abs(high[j] - close[j - 1]),
            abs(low[j] - close[j - 1]),
        )
    atr = float(tr[:n].mean())
    out[n - 1] = atr
    for i in range(n, len(close)):
        atr = (atr * (n - 1) + tr[i]) / n
        out[i] = atr
    return out


def _macd_hist(
    close: np.ndarray,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
    min_bars: int = MACD_MIN_BARS,
) -> np.ndarray:
    """MACD(fast, slow, signal) histogram = MACD line - signal line.

    NaN until ``min_bars`` closes are available (the recursive EMA is
    defined from the first bar, so the cutoff keeps early, unconverged
    values out of the feature set).
    """
    close = np.asarray(close, dtype=float)
    out = np.full(len(close), np.nan)
    if len(close) < min_bars:
        return out
    macd_line = _ema(close, fast) - _ema(close, slow)
    sig = _ema(macd_line, signal)
    out[min_bars - 1 :] = macd_line[min_bars - 1 :] - sig[min_bars - 1 :]
    return out


def _pct_k(close: np.ndarray, k: int) -> np.ndarray:
    """Trailing k-day simple return: close[i]/close[i-k] - 1."""
    close = np.asarray(close, dtype=float)
    out = np.full(len(close), np.nan)
    if len(close) <= k:
        return out
    denom = close[:-k]
    ok = denom > 0
    vals = np.full(len(close) - k, np.nan)
    vals[ok] = close[k:][ok] / denom[ok] - 1.0
    out[k:] = vals
    return out


def _ma_dist(close: np.ndarray, n: int) -> np.ndarray:
    """(close - SMA_n) / SMA_n, NaN until n closes are available."""
    s = pd.Series(np.asarray(close, dtype=float))
    sma = s.rolling(n, min_periods=n).mean()
    dist = (s - sma) / sma
    return dist.where(sma > 0).to_numpy()


def _trailing_realvol(close: np.ndarray, n: int) -> np.ndarray:
    """Sample std (ddof=1) of daily log returns over the trailing n trading
    days, in daily units. NaN until n valid log returns exist."""
    c = np.asarray(close, dtype=float)
    lret = np.full(len(c), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr_all = np.log(c[1:] / c[:-1])
    lr_all[~np.isfinite(lr_all)] = np.nan
    lr_all[~((c[1:] > 0) & (c[:-1] > 0))] = np.nan
    lret[1:] = lr_all
    return pd.Series(lret).rolling(n, min_periods=n).std(ddof=1).to_numpy()


def _vol_z(volume: np.ndarray, n: int) -> np.ndarray:
    """Volume z-score vs the trailing n trading days (inclusive of D).

    (v[D] - mean) / std, ddof=1. Exactly 0.0 when the trailing volume is
    constant (no deviation); NaN until n observations exist.
    """
    s = pd.Series(np.asarray(volume, dtype=float))
    mu = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=1)
    z = (s - mu) / sd
    return z.where(sd != 0.0, 0.0).to_numpy()


def _relvol(volume: np.ndarray, n: int) -> np.ndarray:
    """volume[D] / median(volume[D-n+1..D]); NaN until n observations."""
    s = pd.Series(np.asarray(volume, dtype=float))
    med = s.rolling(n, min_periods=n).median()
    rel = s / med
    return rel.where(med > 0).to_numpy()


def _ticker_price_indicators(g: pd.DataFrame) -> pd.DataFrame:
    """All trailing indicators for one ticker's bar panel (date-sorted)."""
    g = g.sort_values("date").reset_index(drop=True)
    c = g["close"].to_numpy(dtype=float)
    h = g["high"].to_numpy(dtype=float)
    low = g["low"].to_numpy(dtype=float)
    v = g["volume"].to_numpy(dtype=float)
    atr = _wilder_atr(h, low, c)
    with np.errstate(divide="ignore", invalid="ignore"):
        atr_ratio = atr / c
    atr_ratio[~np.isfinite(atr_ratio)] = np.nan
    return pd.DataFrame(
        {
            "ticker": g["ticker"].to_numpy(),
            "date": g["date"].to_numpy(),
            "ret_1d": _pct_k(c, 1),
            "ret_5d": _pct_k(c, 5),
            "ret_20d": _pct_k(c, 20),
            "rsi_14": _wilder_rsi(c),
            "macd_hist": _macd_hist(c),
            "ma_dist_20": _ma_dist(c, 20),
            "ma_dist_50": _ma_dist(c, 50),
            "realvol_5d": _trailing_realvol(c, 5),
            "realvol_20d": _trailing_realvol(c, 20),
            "atr_14": atr_ratio,
            "vol_z_20": _vol_z(v, 20),
            "relvol_20": _relvol(v, 20),
        }
    )


def _trailing_price_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, trading date): trailing price/volume indicators.

    Every value at row date D uses only bars with date <= D (never later).
    Insufficient history -> NaN, never filled.
    """
    required = {"ticker", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"[price-features] prices missing columns {missing}")
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.date
    p = p.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])
    frames = [_ticker_price_indicators(g) for _, g in p.groupby("ticker")]
    return pd.concat(frames, ignore_index=True)


def build_price_features(
    news: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    log=print,
) -> pd.DataFrame:
    """Compute the 14 trailing price/volume features for each news row.

    ``news`` needs columns: url, ticker, published_at (pub_date is derived
    when absent). ``prices`` needs columns: ticker, date, open, high, low,
    close, volume.

    Returns a DataFrame with columns ``["url", "ctx_asof"] +
    PRICE_FEATURE_NAMES``, in the input row order and on the input index.
    ``ctx_asof`` is the latest trading day strictly before the publish
    date (the point-in-time anchor; None where none exists). Every feature
    uses bars on or before ``ctx_asof`` only; insufficient history is NaN
    (never filled, never forward-filled) and the caller decides how to
    handle it.

    The SPY-relative features (``px_vs_spy_5d``, ``px_vs_spy_20d``,
    ``mkt_spy_ret_20d``, ``mkt_spy_vol_20d``) use the latest benchmark
    trading day on or before the as-of day; when the benchmark is absent
    from ``prices`` they stay NaN (logged, not raised).
    """
    news = news.copy()
    missing_news = {"url", "ticker", "published_at"} - set(news.columns)
    if missing_news:
        raise ValueError(f"[price-features] news missing columns {missing_news}")
    if "pub_date" not in news.columns:
        news["pub_date"] = pd.to_datetime(news["published_at"], utc=True).dt.date

    ind = _trailing_price_indicators(prices)
    calendars = {
        t: np.array([_ordinal(d) for d in sub["date"]])
        for t, sub in ind.groupby("ticker")
    }
    tickers = news["ticker"].to_numpy()
    pos = _asof_positions(calendars, tickers, news["pub_date"].to_numpy())

    out = pd.DataFrame(np.nan, index=news.index, columns=PRICE_FEATURE_NAMES)
    ok = pos >= 0
    by_ticker = {
        t: sub.sort_values("date").reset_index(drop=True)
        for t, sub in ind.groupby("ticker")
    }
    # Intermediate trailing returns, needed for the benchmark-relative
    # features below (not part of PRICE_FEATURE_NAMES: ctx_mom5/ctx_mom20
    # already cover the 5d/20d legs, see px_ret_1d note above).
    t_ret5 = np.full(len(news), np.nan)
    t_ret20 = np.full(len(news), np.nan)
    _PX_COLS = [
        "px_ret_1d",
        "px_rsi_14",
        "px_macd_hist",
        "px_ma_dist_20",
        "px_ma_dist_50",
        "px_realvol_5d",
        "px_realvol_20d",
        "px_atr_14",
        "px_vol_z_20",
        "px_relvol_20",
    ]
    _IND_COLS = [
        "ret_1d",
        "rsi_14",
        "macd_hist",
        "ma_dist_20",
        "ma_dist_50",
        "realvol_5d",
        "realvol_20d",
        "atr_14",
        "vol_z_20",
        "relvol_20",
    ]
    if ok.any():
        for ticker, sub in by_ticker.items():
            m = ok & (tickers == ticker)
            if not m.any():
                continue
            rows = sub.iloc[pos[m]]
            out.loc[m, _PX_COLS] = rows[_IND_COLS].to_numpy()
            t_ret5[m] = rows["ret_5d"].to_numpy()
            t_ret20[m] = rows["ret_20d"].to_numpy()

    # Benchmark leg: latest benchmark trading day on or before the as-of
    # day (same searchsorted pattern as ctx_sector_rel5).
    asof_ord = np.full(len(news), -1, dtype=int)
    for ticker, cal in calendars.items():
        m = ok & (tickers == ticker)
        asof_ord[m] = cal[pos[m]]
    if benchmark in by_ticker:
        spy = by_ticker[benchmark]
        spy_cal = np.array([_ordinal(d) for d in spy["date"]])
        s5 = spy["ret_5d"].to_numpy()
        s20 = spy["ret_20d"].to_numpy()
        sv = spy["realvol_20d"].to_numpy()
        idx_ok = np.flatnonzero(ok)
        sp = np.searchsorted(spy_cal, asof_ord[idx_ok], side="right") - 1
        good = sp >= 0
        ii = idx_ok[good]
        out.iloc[ii, out.columns.get_loc("px_vs_spy_5d")] = t_ret5[ii] - s5[sp[good]]
        out.iloc[ii, out.columns.get_loc("px_vs_spy_20d")] = t_ret20[ii] - s20[sp[good]]
        out.iloc[ii, out.columns.get_loc("mkt_spy_ret_20d")] = s20[sp[good]]
        out.iloc[ii, out.columns.get_loc("mkt_spy_vol_20d")] = sv[sp[good]]
    else:
        log(
            f"[px] benchmark {benchmark!r} not in prices: px_vs_spy_5d, "
            "px_vs_spy_20d, mkt_spy_ret_20d, mkt_spy_vol_20d left NaN"
        )

    result = pd.DataFrame({"url": news["url"].to_numpy()}, index=news.index)
    # ctx_asof: the point-in-time anchor, the latest trading day strictly
    # before the publish date (None where no prior trading day exists).
    # Used by validation.point_in_time to prove features predate publication.
    asof_dates = np.full(len(news), None, dtype=object)
    for ticker, cal in calendars.items():
        m = (tickers == ticker) & (pos >= 0)
        if m.any():
            cal_dates = np.array(
                [pd.Timestamp.fromordinal(int(o)).date() for o in cal]
            )
            asof_dates[m] = cal_dates[pos[m]]
    result["ctx_asof"] = asof_dates
    return pd.concat([result, out], axis=1)
