"""Tests for the EVENTVOL splitter integration on a synthetic event frame.

Uses the real signal_lab.validation.splits machinery with the frozen
EVENTVOL config: 5 folds, expanding, horizon 5, embargo 5 trading days,
min_train 50, seed 7. Event rows are sparse (quarterly), as in the real
campaign.
"""

import numpy as np
import pandas as pd

from signal_lab.validation.splits import WalkForwardConfig, make_splits

N_QUARTERS = 18  # 2022-Q1..2026-Q2: all in the dev period by construction
# 20 tickers x 18 quarters = 360 rows: enough for min_train=50 in fold 1.
TICKERS = tuple(f"T{i:02d}" for i in range(20))


def _synthetic_event_frame():
    # Quarterly event dates from 2022-Q1.
    t0s = pd.date_range("2022-01-15", periods=N_QUARTERS, freq="QS")
    t0s = [pd.bdate_range(d, periods=1)[0] for d in t0s]
    rows = []
    for t in TICKERS:
        for d in t0s:
            t0 = d.date()
            rows.append({"ticker": t, "t0": t0,
                         "t1": (d + pd.offsets.BDay(5)).date(),
                         "published_at": pd.Timestamp(t0, tz="UTC"),
                         "target": 0.02})
    f = pd.DataFrame(rows).sort_values(["t0", "ticker"]).reset_index(drop=True)
    return f


def _config():
    return WalkForwardConfig(n_splits=5, window="expanding", horizon_days=5,
                             embargo_days=5, min_train=50, seed=7)


def _trading_ordinals():
    cal = pd.bdate_range("2021-01-01", "2027-01-01")
    return np.array(sorted(d.toordinal() for d in cal.date), dtype=np.int64)


def test_event_splits_five_folds_with_invariants():
    f = _synthetic_event_frame()
    splits = make_splits(f, _config(), _trading_ordinals())
    assert len(splits) == 5
    starts = [s["test_start"] for s in splits]
    ends = [s["test_end"] for s in splits]
    assert starts == sorted(starts)
    for a, b in zip(ends, starts[1:]):
        assert a < b
    for s in splits:
        assert len(s["train_idx"]) >= 50
        assert len(s["test_idx"]) > 0
        # purge: no train row's label window reaches the test range
        tr = f.iloc[s["train_idx"]]
        ts = pd.Timestamp(s["test_start"]).date()
        assert (tr["t1"] < ts).all()
        # no train/test row overlap
        assert not set(s["train_idx"]) & set(s["test_idx"])


def test_event_splits_embargo_zones_empty():
    f = _synthetic_event_frame()
    splits = make_splits(f, _config(), _trading_ordinals())
    cal = _trading_ordinals()
    for k, s in enumerate(splits[1:], start=2):
        prev_end = pd.Timestamp(splits[k - 2]["test_end"]).date().toordinal()
        pos = int(np.searchsorted(cal, prev_end, side="left"))
        zone_end = cal[pos + 5] if pos + 5 < len(cal) else -1
        tr = f.iloc[s["train_idx"]]
        t0o = tr["t0"].apply(lambda d: d.toordinal())
        assert not ((t0o > prev_end) & (t0o <= zone_end)).any()


def test_event_splits_test_rows_within_fold_ranges():
    f = _synthetic_event_frame()
    splits = make_splits(f, _config(), _trading_ordinals())
    seen: set[int] = set()
    for s in splits:
        idx = set(s["test_idx"])
        assert not seen & idx  # disjoint test folds
        seen |= idx
        ts = pd.Timestamp(s["test_start"]).date()
        te = pd.Timestamp(s["test_end"]).date()
        tr = f.iloc[s["test_idx"]]
        assert ((tr["t0"] >= ts) & (tr["t0"] <= te)).all()
