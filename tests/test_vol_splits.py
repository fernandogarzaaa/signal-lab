"""Tests for vol splitter integration on a synthetic vol frame.

Uses the real signal_lab.validation.splits machinery (make_splits,
single_purged_split) with the frozen VOL config: 5 folds, expanding,
horizon 5, embargo 5 trading days, min_train 50, seed 7.
"""

import numpy as np
import pandas as pd

from signal_lab.validation.splits import (
    WalkForwardConfig,
    make_splits,
    single_purged_split,
)

N_DAYS = 250
TICKERS = ("AAA", "BBB", "CCC")


def _synthetic_vol_frame():
    dates = pd.bdate_range("2022-01-03", periods=N_DAYS)
    rows = []
    for t in TICKERS:
        for i, d in enumerate(dates):
            t0 = d.date()
            t1 = dates[i + 5].date() if i + 5 < N_DAYS else None
            rows.append({
                "ticker": t, "t0": t0, "t1": t1,
                "published_at": pd.Timestamp(t0, tz="UTC"),
                "target": 0.01,
            })
    f = pd.DataFrame(rows).dropna(subset=["t1"]).reset_index(drop=True)
    return f, dates


def _config():
    return WalkForwardConfig(n_splits=5, window="expanding", horizon_days=5,
                             embargo_days=5, min_train=50, seed=7)


def _trading_ordinals(dates):
    return np.array(sorted(d.toordinal() for d in dates.date), dtype=np.int64)


def test_make_splits_five_folds_with_invariants():
    f, dates = _synthetic_vol_frame()
    splits = make_splits(f, _config(), _trading_ordinals(dates))
    assert len(splits) == 5
    # test ranges are ordered and non-overlapping
    starts = [s["test_start"] for s in splits]
    ends = [s["test_end"] for s in splits]
    assert starts == sorted(starts)
    for a, b in zip(ends, starts[1:]):
        assert a < b
    for s in splits:
        assert len(s["train_idx"]) >= 50
        assert len(s["test_idx"]) > 0
        # no train row's label window reaches the test range (purge j==k)
        tr = f.iloc[s["train_idx"]]
        ts = pd.Timestamp(s["test_start"]).date()
        assert (tr["t1"] < ts).all()


def test_make_splits_embargo_zones_empty():
    f, dates = _synthetic_vol_frame()
    splits = make_splits(f, _config(), _trading_ordinals(dates))
    cal = np.array(sorted(d.toordinal() for d in dates.date))
    for k, s in enumerate(splits[1:], start=2):
        prev_end = pd.Timestamp(splits[k - 2]["test_end"]).date().toordinal()
        pos = np.searchsorted(cal, prev_end, side="left")
        zone_end = cal[pos + 5] if pos + 5 < len(cal) else -1
        tr = f.iloc[s["train_idx"]]
        t0o = tr["t0"].apply(lambda d: d.toordinal())
        assert not ((t0o > prev_end) & (t0o <= zone_end)).any()


def test_single_purged_split_for_selection():
    f, dates = _synthetic_vol_frame()
    split = single_purged_split(f, _config(), _trading_ordinals(dates),
                                valid_fraction=0.2)
    assert len(split["valid_idx"]) > 0
    assert len(split["train_idx"]) >= 50
    assert len(set(split["train_idx"]) & set(split["valid_idx"])) == 0
    vs = pd.Timestamp(split["valid_start"]).date()
    tr = f.iloc[split["train_idx"]]
    # no train row's [t0, t1] overlaps the validation range
    ve = pd.Timestamp(split["valid_end"]).date()
    overlap = (tr["t0"] <= ve) & (tr["t1"] >= vs)
    assert not overlap.any()
