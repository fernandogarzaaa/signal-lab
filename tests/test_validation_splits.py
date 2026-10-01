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


# ---------------------------------------------------------------------------
# APP-006: purgedcv cross-check edge cases
# (eslazarev/purged-cross-validation v0.1.10, validate_times semantics)
# ---------------------------------------------------------------------------


def test_inverted_label_window_rejected_everywhere():
    # purgedcv's validate_times rejects evaluation < prediction; our purge
    # rule (t1 >= ts_k) would silently misbehave on such rows, so we fail
    # loudly in all three entry points.
    df, td = _frame()
    df.loc[0, "t1"] = pd.Timestamp("2025-12-01").date()  # t1 < t0
    with pytest.raises(ValueError, match="inverted label window"):
        S.make_splits(df, _cfg(), td)
    with pytest.raises(ValueError, match="inverted label window"):
        S.single_purged_split(df, _cfg(), td)
    with pytest.raises(ValueError, match="inverted label window"):
        S.purge_against(
            df["t0"], df["t1"],
            pd.Timestamp("2026-02-01").date(), pd.Timestamp("2026-03-01").date(),
        )


def test_purge_boundary_t1_equals_test_start_is_purged():
    # Conservative boundary, stricter than purgedcv's half-open reference
    # (which keeps t1 == test_start): a train row whose label window ends
    # exactly on the test start is dropped, never kept.
    df, td = _frame()
    splits = S.make_splits(df, _cfg(), td)
    sp = splits[0]
    ts = pd.Timestamp(sp["test_start"])
    t1 = pd.to_datetime(df["t1"].iloc[sp["train_idx"]])
    assert (t1 < ts).all()
    # And the row AT the boundary was among the purged, not the kept.
    assert sp["n_purged"] >= 1


def test_purge_against_boundary_closed_intervals():
    # purge_against uses closed intervals: t1 == test_start overlaps.
    df, _ = _frame(n=20)
    t0 = df["t0"]
    t1 = df["t1"]
    ts = pd.to_datetime(t1.iloc[5]).date()
    te = pd.to_datetime(t1.iloc[10]).date()
    keep = S.purge_against(t0, t1, ts, te)
    # Row 5 has t1 == ts -> dropped (closed-interval overlap).
    assert not keep.iloc[5]
    # A row whose t0 == te overlaps the closed window -> dropped.
    te_rows = pd.to_datetime(t0) == pd.Timestamp(te)
    assert te_rows.any() and not keep[te_rows].any()
    # Rows fully before the window are kept.
    assert keep.iloc[0]


def test_embargo_zone_past_calendar_end_is_skipped():
    # Trading calendar ends right after the last test period: the embargo
    # zone arithmetic must skip (not crash, not assert) when it runs past
    # the calendar end.
    df, _ = _frame(n=60)
    short_td = np.sort(
        np.array(
            [x.toordinal() for x in pd.to_datetime(df["t0"]).dt.date.unique()][:62],
            dtype=np.int64,
        )
    )
    splits = S.make_splits(df, _cfg(), short_td)
    assert len(splits) == 3
    for sp in splits:
        assert sp["n_embargoed"] >= 0


def test_kept_rows_are_conservative_vs_half_open_reference():
    # Independent cross-check against purgedcv's half-open purge semantics:
    # reference drops a train row iff [t0, t1) overlaps [ts_k, te_k),
    # i.e. t0 < te_k and t1 > ts_k. Our rule (drop t1 >= ts_k) must never
    # KEEP a row the reference drops; it may drop more (the t1 == ts_k
    # boundary), which is the documented conservative choice.
    rng = np.random.default_rng(11)
    for trial in range(5):
        n = 90
        days = pd.bdate_range("2026-01-05", periods=n + 8)
        t0 = pd.to_datetime(rng.choice(days[:n], size=n, replace=False))
        horizon = rng.integers(1, 6, size=n)
        t1 = t0 + pd.to_timedelta(horizon, unit="D")
        df = pd.DataFrame(
            {
                "published_at": t0 + pd.Timedelta(hours=15),
                "t0": [x.date() for x in t0],
                "t1": [x.date() for x in t1],
            }
        )
        td = np.sort(
            np.array([x.date().toordinal() for x in days], dtype=np.int64)
        )
        cfg = S.WalkForwardConfig(
            n_splits=3, horizon_days=8, embargo_days=8, min_train=5
        )
        for sp in S.make_splits(df, cfg, td):
            ts = pd.Timestamp(sp["test_start"]).toordinal()
            te = pd.Timestamp(sp["test_end"]).toordinal()
            kept = df.iloc[sp["train_idx"]]
            kt0 = pd.to_datetime(kept["t0"]).map(pd.Timestamp.toordinal)
            kt1 = pd.to_datetime(kept["t1"]).map(pd.Timestamp.toordinal)
            # Reference would drop rows with kt0 < te and kt1 > ts.
            ref_dropped = (kt0 < te) & (kt1 > ts)
            assert not ref_dropped.any(), (
                f"trial {trial} fold {sp['fold']}: kept row the half-open "
                "reference would purge"
            )
