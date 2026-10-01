"""Per-fold label construction (Phase 1).

The label *value* (abn_ret, the forward abnormal return) is a per-article
fact and carries no leakage. The label *cutoff* (top-decile threshold) and
the attention-filter quantiles are statistics of a sample, and computing
them on the full dataset before splitting leaks test information into
train labels (audit findings 4c, 4d). This module computes them on train
data only:

- ``fold_cutoff(abn_ret_train, quantile)``: the cutoff for one fold.
- ``fold_labels(abn_ret, cutoff)``: binary labels from a cutoff.
- ``attention_keep_mask(train, full, ...)``: the attention rule, with the
  per-ticker day-count distribution estimated on TRAIN rows only and
  applied to every row of the fold.

The legacy global-cutoff ``models.labels.build_labels`` is unchanged for
CLI compatibility; no validation path uses it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fold_cutoff(abn_ret_train: pd.Series, quantile: float) -> float:
    """Top-quantile cutoff from TRAIN abnormal returns only."""
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    v = pd.to_numeric(abn_ret_train, errors="coerce").dropna()
    if len(v) == 0:
        raise ValueError("[labels] no usable train abnormal returns for cutoff")
    return float(v.quantile(quantile))


def fold_labels(abn_ret: pd.Series, cutoff: float) -> pd.Series:
    """Binary labels: 1 iff abn_ret >= cutoff."""
    return (pd.to_numeric(abn_ret, errors="coerce") >= cutoff).astype(int)


def _qrank_against(count: int, train_counts: np.ndarray) -> float:
    """Fraction of the train day-counts strictly quieter than ``count``."""
    if len(train_counts) == 0:
        return 0.0
    return float((train_counts < count).mean())


def attention_keep_mask(
    train: pd.DataFrame,
    full: pd.DataFrame,
    min_articles: int = 2,
    top_quartile: bool = True,
) -> pd.Series:
    """Attention keep-mask with train-only statistics.

    ``train`` and ``full`` need columns ``ticker``, ``pub_date``. The
    per-ticker daily article-count distribution is estimated on ``train``
    rows only; each row of ``full`` is kept when its ticker-day count is
    >= ``min_articles`` or (``top_quartile``) its train-estimated qrank is
    >= 0.75. Rows whose ticker never appears in train get qrank 0.0 (the
    min_articles branch can still keep them).
    """
    if min_articles < 1:
        raise ValueError("min_articles must be >= 1")
    for name, frame in (("train", train), ("full", full)):
        missing = {"ticker", "pub_date"} - set(frame.columns)
        if missing:
            raise ValueError(f"[labels] {name} frame missing columns {missing}")

    train_counts = (
        train.groupby(["ticker", "pub_date"]).size().rename("n").reset_index()
    )
    by_ticker: dict[str, np.ndarray] = {
        t: g["n"].to_numpy() for t, g in train_counts.groupby("ticker")
    }
    full_counts = (
        full.groupby(["ticker", "pub_date"]).size().rename("n").reset_index()
    )

    def _keep(row) -> bool:
        c = int(row["n"])
        if c >= min_articles:
            return True
        if top_quartile:
            q = _qrank_against(c, by_ticker.get(row["ticker"], np.array([])))
            return q >= 0.75
        return False

    keep_day = full_counts.apply(_keep, axis=1)
    keep_map = {
        (r["ticker"], r["pub_date"]): k
        for (_, r), k in zip(full_counts.iterrows(), keep_day)
    }
    mask = [
        keep_map.get((t, d), False)
        for t, d in zip(full["ticker"], full["pub_date"])
    ]
    return pd.Series(mask, index=full.index)
