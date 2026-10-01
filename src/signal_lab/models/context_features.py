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
