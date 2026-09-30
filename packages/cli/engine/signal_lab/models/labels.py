"""Weak-label construction for the stage-3 sentiment classifier (0.3.0 WS1).

Two labeling schemes, selectable at runtime:

- ``"baseline"`` (the 0.2.0 scheme, kept for comparison): label = 1 iff the
  ticker's next-trading-day abnormal return vs the market benchmark lands in
  the top decile across the sample.
- ``"windowed"`` (0.3.0 default): label = 1 iff the ticker's *cumulative*
  abnormal return over trading days t+1..t+window_days lands in the top
  decile, and the article's ticker-day passes the attention filter (the day
  carried enough news presence that its move is plausibly news-related).

Abnormal return is always ticker return minus market return over the same
forward window. The market leg is the benchmark ticker's return; when the
benchmark has no price on a given date the universe equal-weight mean fills
in (same fallback the 0.2.0 code used).

Both schemes return ``(labeled_df, label_config)``. ``label_config`` is a
plain-JSON dict that stage 3 carries verbatim in its output, so runs under
different labelings stay comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SCHEMES = ("windowed", "baseline")


@dataclass(frozen=True)
class LabelConfig:
    """Parameters controlling weak-label construction."""

    scheme: str = "windowed"
    window_days: int = 3
    quantile: float = 0.90
    benchmark: str = "SPY"
    attention_enabled: bool = True
    attention_min_articles: int = 2
    attention_top_quartile: bool = True

    def validated(self) -> LabelConfig:
        if self.scheme not in SCHEMES:
            raise ValueError(
                f"unknown labeling scheme {self.scheme!r}; expected one of {SCHEMES}"
            )
        if not isinstance(self.window_days, int) or self.window_days < 1:
            raise ValueError(
                f"window_days must be a positive int, got {self.window_days!r}"
            )
        if not 0.0 < self.quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {self.quantile!r}")
        if not self.benchmark:
            raise ValueError("benchmark ticker must be a non-empty string")
        if self.attention_min_articles < 1:
            raise ValueError("attention_min_articles must be >= 1")
        return self


def baseline_config() -> LabelConfig:
    """The 0.2.0 labeling, exactly: t+1 top-decile, no attention filter."""
    return LabelConfig(
        scheme="baseline", window_days=1, attention_enabled=False
    ).validated()


def _daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, date): close-to-close return vs the previous trading day."""
    p = prices.sort_values(["ticker", "date"]).copy()
    p["ret"] = p.groupby("ticker")["close"].pct_change()
    return p[["ticker", "date", "ret"]]


def _market_daily_returns(rets: pd.DataFrame, benchmark: str) -> pd.Series:
    """Per-date market return: benchmark return, else the universe
    equal-weight mean return for that date."""
    bench = (
        rets[rets.ticker == benchmark].drop_duplicates("date").set_index("date")["ret"]
    )
    umean = rets[rets.ticker != benchmark].groupby("date")["ret"].mean()
    dates = rets["date"].drop_duplicates().sort_values()
    mkt = bench.reindex(dates)
    return mkt.fillna(umean.reindex(dates)).rename("mkt_ret")


def forward_cumulative_abnormal(
    prices: pd.DataFrame, config: LabelConfig
) -> pd.DataFrame:
    """Per (ticker, date): cumulative abnormal return over the next
    ``window_days`` trading days.

    Returns columns: ticker, date, ticker_fwd_ret, mkt_fwd_ret, abn_ret,
    fwd_days (how many forward trading days had prices). With
    ``window_days=1`` this is exactly the 0.2.0 next-day abnormal return.
    """
    config = config.validated()
    rets = _daily_returns(prices)
    mkt = _market_daily_returns(rets, config.benchmark)
    rets = rets.merge(mkt, on="date", how="left")

    cum_t = pd.Series(1.0, index=rets.index)
    cum_m = pd.Series(1.0, index=rets.index)
    fwd_days = pd.Series(0, index=rets.index)
    for k in range(1, config.window_days + 1):
        rt = rets.groupby("ticker")["ret"].shift(-k)
        rm = rets.groupby("ticker")["mkt_ret"].shift(-k)
        ok = rt.notna() & rm.notna()
        cum_t[ok] = cum_t[ok] * (1.0 + rt[ok])
        cum_m[ok] = cum_m[ok] * (1.0 + rm[ok])
        fwd_days = fwd_days + ok.astype(int)

    out = rets[["ticker", "date"]].copy()
    out["ticker_fwd_ret"] = cum_t - 1.0
    out["mkt_fwd_ret"] = cum_m - 1.0
    out["abn_ret"] = out["ticker_fwd_ret"] - out["mkt_fwd_ret"]
    out["fwd_days"] = fwd_days.astype(int)
    return out


def apply_attention(
    df: pd.DataFrame, min_articles: int = 2, top_quartile: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Keep only ticker-days with sufficient news presence.

    A ticker-day passes when it has at least ``min_articles`` distinct
    articles, OR (when ``top_quartile``) its article count is strictly
    greater than at least 75% of that ticker's own daily article counts.
    The per-ticker comparison is computed over distinct ticker-days, not
    over articles, so busy days do not inflate their own threshold, and
    the strict inequality keeps tied quiet days from passing on each
    other's backs.

    ``df`` must have columns ticker, pub_date, url. Returns
    (filtered_df, info) where info reports dropped_rows / dropped_pct and the
    kept frame carries an ``n_articles`` column (that day's article count).
    """
    if min_articles < 1:
        raise ValueError("min_articles must be >= 1")
    day_counts = (
        df.groupby(["ticker", "pub_date"]).size().rename("n_articles").reset_index()
    )
    if top_quartile:
        # qrank: fraction of the ticker's days strictly quieter than this day
        def _qrank(s: pd.Series) -> np.ndarray:
            v = s.to_numpy()
            return (v[:, None] > v[None, :]).mean(axis=1)

        day_counts["qrank"] = day_counts.groupby("ticker")["n_articles"].transform(
            _qrank
        )
        day_counts["keep"] = (day_counts["n_articles"] >= min_articles) | (
            day_counts["qrank"] >= 0.75
        )
    else:
        day_counts["keep"] = day_counts["n_articles"] >= min_articles
    n_before = len(df)
    df = df.merge(
        day_counts[["ticker", "pub_date", "n_articles", "keep"]],
        on=["ticker", "pub_date"],
        how="left",
    )
    dropped = int((~df["keep"]).sum())
    kept = df[df["keep"]].drop(columns=["keep"]).reset_index(drop=True)
    info = {
        "dropped_rows": dropped,
        "dropped_pct": round(dropped / n_before, 4) if n_before else 0.0,
        "min_articles": min_articles,
        "top_quartile": bool(top_quartile),
    }
    return kept, info


def label_config_block(
    df: pd.DataFrame,
    config: LabelConfig,
    cutoff: float,
    attention_info: dict | None,
    partial_dropped: int,
) -> dict:
    """Plain-JSON dict describing the labeling behind a labeled dataframe.

    This is the ``label_config`` block stage 3 emits verbatim.
    """
    attn = (
        dict(attention_info)
        if attention_info
        else {
            "dropped_rows": 0,
            "dropped_pct": 0.0,
            "min_articles": config.attention_min_articles,
            "top_quartile": config.attention_top_quartile,
        }
    )
    attn["enabled"] = bool(config.attention_enabled)
    return {
        "scheme": config.scheme,
        "window_days": config.window_days,
        "quantile": float(config.quantile),
        "benchmark": config.benchmark,
        "attention": attn,
        "cutoff": round(float(cutoff), 6),
        "positive_rate": round(float(df["label"].mean()), 4),
        "n_labeled": len(df),
        "partial_window_rows_dropped": int(partial_dropped),
    }


def build_labels(
    news: pd.DataFrame, prices: pd.DataFrame, config: LabelConfig
) -> tuple[pd.DataFrame, dict]:
    """Attach weak labels to news articles.

    ``news`` needs columns: url, published_at, pub_date, ticker (ticker
    extraction happens upstream in build_and_train). ``prices`` needs
    columns: ticker, date, close.

    Returns (labeled_df, label_config). labeled_df columns: url, ticker,
    published_at, pub_date, fwd_days, n_articles (0 when the attention
    filter is off), ticker_fwd_ret, mkt_fwd_ret, abn_ret, label.
    """
    config = config.validated()
    fwd = forward_cumulative_abnormal(prices, config)
    df = news.merge(
        fwd, left_on=["ticker", "pub_date"], right_on=["ticker", "date"], how="left"
    )
    df = df.dropna(subset=["abn_ret"]).reset_index(drop=True)
    partial_dropped = int((df["fwd_days"] < config.window_days).sum())
    df = df[df["fwd_days"] >= config.window_days].reset_index(drop=True)

    attention_info = None
    if config.attention_enabled:
        df, attention_info = apply_attention(
            df,
            min_articles=config.attention_min_articles,
            top_quartile=config.attention_top_quartile,
        )
    else:
        df["n_articles"] = 0

    if df.empty:
        raise ValueError(
            "[labels] no labeled rows: none of the news articles could be "
            "matched to a full forward price window (articles may be newer "
            "than the latest prices, or entity extraction found no tickers). "
            "Ingest an older news window before training."
        )
    cutoff = float(df["abn_ret"].quantile(config.quantile))
    df["label"] = (df["abn_ret"] >= cutoff).astype(int)
    block = label_config_block(df, config, cutoff, attention_info, partial_dropped)
    cols = [
        "url",
        "ticker",
        "published_at",
        "pub_date",
        "fwd_days",
        "n_articles",
        "ticker_fwd_ret",
        "mkt_fwd_ret",
        "abn_ret",
        "label",
    ]
    return df[cols].reset_index(drop=True), block
