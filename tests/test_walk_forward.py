"""Unit tests for purged walk-forward CV (0.3.0 WS5). No DB, no network."""

import numpy as np
import pandas as pd
import pytest

from signal_lab.models.walk_forward import run_walk_forward, walk_forward_splits


def _frame(n=120, start="2026-01-01"):
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=n, freq="D", tz="UTC")
    texts = ["apple stock rises on earnings beat" if i % 3 else "market flat today" for i in range(n)]
    labels = (rng.random(n) < 0.15).astype(int)
    return pd.DataFrame({"published_at": dates, "text": texts, "label": labels})


def test_splits_no_train_test_overlap():
    df = _frame()
    splits = walk_forward_splits(df["published_at"], n_splits=5, min_train=10)
    assert len(splits) == 5
    for sp in splits:
        assert len(set(sp["train_idx"]) & set(sp["test_idx"])) == 0
        # train strictly before test in time (positional cut)
        assert df["published_at"].iloc[sp["train_idx"]].max() <= df["published_at"].iloc[sp["test_idx"]].min()


def test_purge_drops_label_window_overlap():
    # Daily articles; purge_days=5 must drop the 5 train days before test start.
    df = _frame(n=120)
    splits = walk_forward_splits(df["published_at"], n_splits=5, purge_days=5, embargo_days=0, min_train=10)
    sp = splits[0]
    test_start = pd.to_datetime(sp["test_start"])
    train_times = df["published_at"].iloc[sp["train_idx"]]
    assert (train_times <= test_start - pd.Timedelta(days=5)).all()
    assert sp["n_purged"] > 0


def test_embargo_drops_post_test_start_rows():
    df = _frame(n=200)
    splits = walk_forward_splits(df["published_at"], n_splits=5, purge_days=0, embargo_days=7, min_train=10)
    # Fold 2 trains on blocks 0..1; embargo must remove block-1 rows
    # published within 7 days after block 1 (second test block) started.
    sp = splits[1]
    first_test_start = pd.to_datetime(splits[0]["test_start"])
    train_times = df["published_at"].iloc[sp["train_idx"]]
    embargoed_zone = (train_times > first_test_start) & (
        train_times <= first_test_start + pd.Timedelta(days=7)
    )
    assert not embargoed_zone.any()


def test_run_walk_forward_structure_and_aggregates():
    df = _frame(n=150)
    result = run_walk_forward(df, n_splits=3, log=lambda *a, **k: None, purge_days=2, embargo_days=2, min_train=10)
    assert result["n_splits"] == 3
    assert len(result["folds"]) == 3
    for f in result["folds"]:
        assert f["n_train"] > 0 and f["n_test"] > 0
        assert 0.0 <= f["pr_auc"] <= 1.0
        assert f["n_train_pos"] + f["n_test_pos"] <= len(df)
    agg = result["aggregate"]["pr_auc"]
    assert agg["ci95"][0] <= agg["mean"] <= agg["ci95"][1]
    assert agg["std"] >= 0


def test_run_walk_forward_deterministic():
    df = _frame(n=150)
    kw = dict(n_splits=3, log=lambda *a, **k: None, purge_days=2, embargo_days=2, min_train=10)
    r1 = run_walk_forward(df, **kw)
    r2 = run_walk_forward(df, **kw)
    assert r1["folds"] == r2["folds"]
    assert r1["aggregate"] == r2["aggregate"]


def test_run_walk_forward_with_dense_extra():
    df = _frame(n=150)
    extra = pd.DataFrame(
        {"ctx_vol20_pct": np.random.default_rng(1).random(150)},
    )
    result = run_walk_forward(df, dense_extra=extra, n_splits=3, log=lambda *a, **k: None, purge_days=2, embargo_days=2, min_train=10)
    assert len(result["folds"]) == 3


def test_min_train_guards_degenerate_folds():
    df = _frame(n=30)
    with pytest.raises(ValueError, match="min_train"):
        walk_forward_splits(df["published_at"], n_splits=5, min_train=10**9)


def test_single_class_fold_is_skipped_loudly():
    # All-negative early block: fold 1's train has one class and must be
    # recorded as skipped, not crash or fabricate a metric.
    df = _frame(n=150)
    df.loc[df.index[:60], "label"] = 0
    result = run_walk_forward(
        df, n_splits=3, purge_days=2, embargo_days=2, min_train=10,
        log=lambda *a, **k: None,
    )
    assert len(result["skipped_folds"]) >= 1
    assert all("train classes" in s["reason"] for s in result["skipped_folds"])
    assert len(result["folds"]) + len(result["skipped_folds"]) == 3
