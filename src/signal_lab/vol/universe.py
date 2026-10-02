"""Frozen VOL universe: S&P 100 constituents + SPY.

Source: Wikipedia "S&P 100" components table, "as of September 21, 2026",
retrieved 2026-10-02 (pre-campaign). The committed artifact
``data/universe_vol.csv`` is the frozen universe; this module embeds the
same list so the CSV can be regenerated and verified bit-for-bit.

Count: 101 constituents (both Alphabet share classes) + SPY = 102.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Canonical tickers as listed by the source. BRK.B is the S&P listing
# spelling; yfinance wants "BRK-B" (see YFINANCE_OVERRIDES).
SP100_TICKERS: list[str] = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMAT", "AMD", "AMGN", "AMT",
    "AMZN", "ANET", "AVGO", "AXP", "BA", "BAC", "BKNG", "BLK", "BMY",
    "BNY", "BRK.B", "C", "CAT", "CMCSA", "COF", "COP", "COST", "CRM",
    "CSCO", "CVS", "CVX", "DE", "DELL", "DHR", "DIS", "DUK", "EMR",
    "FDX", "GD", "GE", "GEV", "GILD", "GM", "GOOG", "GOOGL", "GS",
    "HD", "IBM", "INTC", "INTU", "ISRG", "JNJ", "JPM", "KO", "LIN",
    "LLY", "LMT", "LOW", "LRCX", "MA", "MCD", "MDLZ", "MDT", "META",
    "MMM", "MO", "MRK", "MS", "MSFT", "MU", "NEE", "NFLX", "NOW",
    "NVDA", "ORCL", "PANW", "PEP", "PFE", "PG", "PLTR", "PM", "QCOM",
    "RTX", "SBUX", "SCHW", "SNDK", "SO", "T", "TMO", "TMUS", "TSLA",
    "TXN", "UBER", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC",
    "WMT", "XOM",
]

MARKET_LEG = "SPY"

# Canonical -> yfinance spelling.
YFINANCE_OVERRIDES: dict[str, str] = {"BRK.B": "BRK-B"}


def universe_tickers() -> list[str]:
    """Frozen universe: 101 S&P 100 constituents + SPY (102 total)."""
    tickers = list(SP100_TICKERS) + [MARKET_LEG]
    assert len(SP100_TICKERS) == 101, f"S&P 100 list has {len(SP100_TICKERS)} entries, want 101"
    assert len(tickers) == 102, f"universe has {len(tickers)} entries, want 102"
    assert len(set(tickers)) == len(tickers), "duplicate tickers in universe"
    return tickers


def yfinance_ticker(canonical: str) -> str:
    """Map a canonical ticker to its yfinance spelling."""
    return YFINANCE_OVERRIDES.get(canonical, canonical)


def write_universe_csv(path: str | Path) -> Path:
    """Write the frozen universe CSV (single ``ticker`` column)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": universe_tickers()}).to_csv(path, index=False)
    return path


def load_universe(path: str | Path) -> list[str]:
    """Load tickers from a committed universe CSV; fail loud on mismatch."""
    df = pd.read_csv(path)
    if list(df.columns) != ["ticker"]:
        raise ValueError(f"universe CSV must have a single 'ticker' column, got {list(df.columns)}")
    tickers = df["ticker"].astype(str).tolist()
    if tickers != universe_tickers():
        raise ValueError(
            "universe CSV does not match the frozen S&P 100 + SPY list; "
            "the committed CSV is frozen and must not drift"
        )
    return tickers
