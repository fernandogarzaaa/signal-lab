"""Same-ticker same-t0 dedupe / down-weighting (Phase 1).

Articles sharing a (ticker, t0) share a label window, so their labels are
perfectly correlated. Treating them as independent samples overstates the
effective sample size and lets busy news days dominate training.

Two policies (configurable; the runner defaults to ``"weight"``):

- ``"first"``: keep the earliest-published article per (ticker, t0).
- ``"weight"``: keep all rows; return ``sample_weight = 1 / group_size``
  so each (ticker, t0) contributes one effective sample to the fit.

Group sizes are counts of articles, a per-day fact with no label or
future information, so computing them on the full fold is safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def group_sizes(df: pd.DataFrame) -> pd.Series:
    """Per-row size of its (ticker, t0) group."""
    missing = {"ticker", "t0"} - set(df.columns)
    if missing:
        raise ValueError(f"[dedupe] frame missing columns {missing}")
    return df.groupby(["ticker", df["t0"].astype(str)])["ticker"].transform("size")


def dedupe_first(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the earliest-published article per (ticker, t0)."""
    missing = {"ticker", "t0", "published_at"} - set(df.columns)
    if missing:
        raise ValueError(f"[dedupe] frame missing columns {missing}")
    work = df.sort_values("published_at", kind="stable")
    key = work["ticker"].astype(str) + "|" + work["t0"].astype(str)
    return work[~key.duplicated(keep="first")].sort_index()


def sample_weights(df: pd.DataFrame) -> np.ndarray:
    """1 / (ticker, t0) group size per row (sums to n_groups)."""
    sizes = group_sizes(df).to_numpy(dtype=float)
    return 1.0 / sizes


def apply(df: pd.DataFrame, policy: str = "weight") -> tuple[pd.DataFrame, np.ndarray | None]:
    """Apply a dedupe policy. Returns (frame, sample_weight or None)."""
    if policy == "first":
        return dedupe_first(df), None
    if policy == "weight":
        return df, sample_weights(df)
    raise ValueError(f"unknown dedupe policy {policy!r}; expected 'first'|'weight'")
