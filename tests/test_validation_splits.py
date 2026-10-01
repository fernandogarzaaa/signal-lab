"""Tests for validation.splits: purged + embargoed walk-forward invariants.

The splitter is the load-bearing wall of the honest protocol. These tests
pin the exact purge/embargo semantics and the runtime assertions:
  - no train/test index overlap, train t0 strictly before test start
  - no kept train row has t1 >= the fold's test start (exact purge)
  - embargo zones after earlier test periods are empty
  - purge/embargo only ever REMOVE rows (never add or move)
  - determinism, rolling-window bounds, retrain_every, min_train guard
"""

import numpy as np
import pandas as pd
import pytest

from signal_lab.validation import periods, splits as S


def _frame(n=120, start="2026-01-05", seed=0, horizon=3):
    """One article per trading day, t1 = t0 + ``horizon`` trading days."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n + horizon)
    d = [x.date() for x in days[:n]]
    df = pd.DataFrame(
        {
            "published_at": pd.to_datetime(
                [x + pd.Timedelta(hours=15) for x in days[:n]], utc=True
            ),
            "t0": d,
            "t1": [x.date() for x in days[horizon : n + horizon]],
            "abn_ret": rng.normal(0, 0.02, n),
        }
    )
    trading_days = np.sort(
        np.array([x.date().toordinal() for x in days], dtype=np.int64)
    )
    return df, trading_days


def _cfg(**kw):
    base = dict(n_splits=3, horizon_days=3, embargo_days=5, min_train=10)
    base.update(kw)
    return S.WalkForwardConfig(**base)


def test_no_index_overlap_and_time_order():
    df, td = _frame()
    for sp in S.make_splits(df, _cfg(), td):
        tri = set(sp["train_idx"].tolist())
        tei = set(sp["test_idx"].tolist())
        assert tri.isdisjoint(tei)
        t0 = pd.to_datetime(df["t0"])
        assert (t0.iloc[sorted(tri)] < pd.Timestamp(sp["test_start"])).all()


def test_purge_drops_train_rows_with_t1_past_test_start():
    df, td = _frame()
    for sp in S.make_splits(df, _cfg(), td):
        tri = sp["train_idx"]
        t1 = pd.to_datetime(df["t1"].iloc[tri])
        assert (t1 < pd.Timestamp(sp["test_start"])).all(), (
            f"fold {sp['fold']}: kept train rows reach into the test period"
        )


def test_purge_zone_width_matches_horizon():
    # One article per trading day, t1 = t0 + 3: exactly the last 3 train
    # rows of each fold's final train block must be purged (purge is
    # applied before embargo, so the count is exact).
    df, td = _frame()
    for sp in S.make_splits(df, _cfg(), td):
        assert sp["n_purged"] == 3, (sp["fold"], sp["n_purged"])


def test_expanding_window_grows():
    df, td = _frame()
    sizes = [len(sp["train_idx"]) for sp in S.make_splits(df, _cfg(), td)]
    assert sizes == sorted(sizes) and len(set(sizes)) > 1, sizes


def test_rolling_window_bounds_train_start():
    df, td = _frame()
    cfg = _cfg(window="rolling", train_window_days=30)
    for sp in S.make_splits(df, cfg, td):
        t0 = pd.to_datetime(df["t0"].iloc[sp["train_idx"]])
        lo = pd.Timestamp(sp["test_start"]) - pd.Timedelta(days=30)
        assert (t0 >= lo).all()


def test_embargo_zone_is_empty_after_earlier_tests():
    df, td = _frame()
    cfg = _cfg(embargo_days=5)
    splits = S.make_splits(df, cfg, td)
    # Re-derive each fold's embargo zones from the returned test ranges.
    for k, sp in enumerate(splits):
        tri = sp["train_idx"]
        t0 = pd.to_datetime(df["t0"].iloc[tri])
        for j in range(k):
            te_j = pd.Timestamp(splits[j]["test_end"])
            # zone_end in trading days; approximate with 7 calendar days
            # (5 trading days <= 7 calendar days, so this is a superset:
            # emptiness on the superset implies emptiness of the zone).
            zone = (t0 > te_j) & (t0 <= te_j + pd.Timedelta(days=7))
            assert not zone.any(), (sp["fold"], j)


def test_purge_embargo_accounting_identity():
    # Rows are only ever removed: purged + embargoed + kept == full block.
    df, td = _frame()
    n = len(df)
    cuts = [int(round(n * k / 4)) for k in range(5)]
    for k, sp in enumerate(S.make_splits(df, _cfg(), td), start=1):
        n_full = cuts[k]  # expanding window: rows [0, b_k)
        assert (
            len(sp["train_idx"]) + sp["n_purged"] + sp["n_embargoed"] == n_full
        ), (k, sp)


def test_deterministic_across_runs():
    df, td = _frame()
    a = S.make_splits(df, _cfg(), td)
    b = S.make_splits(df, _cfg(), td)
    for sa, sb in zip(a, b):
        assert (sa["train_idx"] == sb["train_idx"]).all()
        assert (sa["test_idx"] == sb["test_idx"]).all()


def test_retrain_every_skips_refits():
    df, td = _frame()
    splits = S.make_splits(df, _cfg(retrain_every=2), td)
    assert [s["refit"] for s in splits] == [True, False, True]


def test_min_train_raises_on_degenerate_fold():
    df, td = _frame(n=20)
    with pytest.raises(ValueError, match="min_train"):
        S.make_splits(df, _cfg(min_train=10**9), td)


def test_embargo_can_be_minimal():
    df, td = _frame()
    splits = S.make_splits(df, _cfg(embargo_days=3), td)
    assert all(s["n_embargoed"] >= 0 for s in splits)


def test_purge_false_keeps_overlapping_rows_for_canary():
    # Escape hatch for the leakage canary ONLY: with purge=False the
    # rows whose t1 reaches into the test period are kept, and the
    # runtime assertion is skipped (the canary asserts they ARE kept).
    df, td = _frame()
    cfg = _cfg(purge=False)
    splits = S.make_splits(df, cfg, td)
    sp = splits[0]
    t1 = pd.to_datetime(df["t1"].iloc[sp["train_idx"]])
    assert (t1 >= pd.Timestamp(sp["test_start"])).any()
    assert sp["n_purged"] == 0


def test_single_purged_split_purges_validation_overlap():
    df, td = _frame()
    sp = S.single_purged_split(df, _cfg(), td, valid_fraction=0.2)
    vt0 = pd.to_datetime(df["t0"].iloc[sp["valid_idx"]])
    v_start, v_end = vt0.min(), vt0.max()
    t0 = pd.to_datetime(df["t0"].iloc[sp["train_idx"]])
    t1 = pd.to_datetime(df["t1"].iloc[sp["train_idx"]])
    overlap = (t0 <= v_end) & (t1 >= v_start)
    assert not overlap.any()
    assert sp["n_purged"] > 0
    # Validation block is the LAST 20% of dev rows by t0.
    assert (vt0 >= t0.max()).all()


def test_confirmation_rows_rejected():
    df, td = _frame()
    # Push one row into the confirmation period (t0 >= 2026-09-01).
    df.loc[0, "t0"] = pd.Timestamp("2026-09-15").date()
    with pytest.raises(periods.ConfirmationLeakError):
        S.make_splits(df, _cfg(), td)
