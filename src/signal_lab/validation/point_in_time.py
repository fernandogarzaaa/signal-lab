"""Point-in-time declarations for every feature (Phase 1).

Each feature declares three timestamps:

- ``observation``: when the underlying event happened.
- ``publication``: when the value became knowable.
- ``availability``: when the pipeline may use it (must be strictly before
  the prediction time of every row it serves).

The structural rule for this codebase: price features are computed as of
the latest trading day *strictly before* the article's publication date,
and article metadata comes from the article row itself. ``check_frame``
asserts the rule per row from the stored ``ctx_asof`` date
(latest price-trading-day < pub_date); any violation raises instead of
training on it.
"""

from __future__ import annotations

import pandas as pd

# Feature -> (observation, publication, availability). All price features
# share the same timing; they differ only in what they measure.
_PRICE_TIMING = {
    "observation": "daily closes through the as-of trading day D "
    "(latest trading day strictly before the article's pub_date)",
    "publication": "16:00 ET on day D (the close)",
    "availability": "after the close of D; usable for any article with "
    "pub_date > D, i.e. strictly before prediction time",
}

FEATURE_POINT_IN_TIME: dict[str, dict[str, str]] = {
    "ctx_vol20_pct": {
        **_PRICE_TIMING,
        "measures": "percentile of trailing-20d realized vol vs its own 252d history",
    },
    "ctx_mom5": {
        **_PRICE_TIMING,
        "measures": "trailing 5-trading-day return ending on D",
    },
    "ctx_mom20": {
        **_PRICE_TIMING,
        "measures": "trailing 20-trading-day return ending on D",
    },
    "ctx_sector_rel5": {
        **_PRICE_TIMING,
        "measures": "ticker 5d return minus benchmark/ETF-leg 5d return, same window",
    },
    "ctx_src_tier": {
        "observation": "article URL (known at ingest)",
        "publication": "article publication time",
        "availability": "at prediction time (metadata of the article itself)",
        "measures": "source-domain reputation tier 2/1/0",
    },
    "ctx_word_count": {
        "observation": "article text (known at ingest)",
        "publication": "article publication time",
        "availability": "at prediction time (metadata of the article itself)",
        "measures": "word count of title + snippet",
    },
    "ctx_hour_bucket": {
        "observation": "article publication timestamp (known at ingest)",
        "publication": "article publication time",
        "availability": "at prediction time (metadata of the article itself)",
        "measures": "UTC publish hour in quarters of the day",
    },
}


class PointInTimeViolation(RuntimeError):
    """A feature's availability time is not strictly before prediction time."""


def check_frame(df: pd.DataFrame, context: str = "point_in_time") -> None:
    """Assert every row's price-feature as-of day is strictly before its
    publication date.

    ``df`` needs ``ctx_asof`` (date) and ``published_at``. Raises
    ``PointInTimeViolation`` on the first violation.
    """
    missing = {"ctx_asof", "published_at"} - set(df.columns)
    if missing:
        raise ValueError(f"[{context}] frame missing columns {missing}")
    asof = pd.to_datetime(df["ctx_asof"]).dt.date
    pub = pd.to_datetime(df["published_at"], utc=True).dt.date
    bad = ~(asof < pub) & asof.notna()
    if bool(bad.any()):
        idx = df.index[bad][0]
        raise PointInTimeViolation(
            f"[{context}] row {idx}: ctx_asof={asof.loc[idx]} is not strictly "
            f"before pub_date={pub.loc[idx]}; price features would use "
            "information from the publication day or later."
        )


def describe() -> dict:
    """JSON-safe copy of the registry (for provenance blocks)."""
    return {k: dict(v) for k, v in FEATURE_POINT_IN_TIME.items()}
