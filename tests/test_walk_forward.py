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


def test_folds_carry_roc_auc_ranking_metric():
    """Cross-sectional ranking under purged walk-forward: every scored
    fold reports ROC-AUC (P(random positive outranks random negative)),
    the ranking analogue of PR-AUC. Perfect ranking -> 1.0."""
    rng = np.random.default_rng(3)
    n = 200
    dates = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    # Text perfectly separates the label: ranking must be (near) perfect.
    texts = ["great earnings beat profit surge"] * n
    labels = (rng.random(n) < 0.2).astype(int)
    texts = ["great earnings beat profit surge" if l else "market flat today"
             for l in labels]
    df = pd.DataFrame({"published_at": dates, "text": texts, "label": labels})
    result = run_walk_forward(
        df, n_splits=3, purge_days=2, embargo_days=2, min_train=10,
        log=lambda *a, **k: None,
    )
    assert result["folds"], "expected scored folds"
    for f in result["folds"]:
        assert "roc_auc" in f
        assert 0.5 <= f["roc_auc"] <= 1.0
    agg = result["aggregate"]["roc_auc"]
    assert agg["mean"] > 0.9, f"separable data should rank well, got {agg['mean']}"
    assert "ci95" in agg


def test_provenance_block_records_run_context():
    """build_provenance captures classifier, representation, label scheme,
    code revision, and dataset shape: numbers without this are not
    comparable across runs."""
    from signal_lab.models.labels import LabelConfig
    from signal_lab.models.walk_forward import build_provenance

    df = _frame(n=60)
    cfg = LabelConfig().validated()
    prov = build_provenance("tfidf", cfg, df, log=lambda *a, **k: None)
    assert prov["classifier"] == "logreg_balanced"
    assert prov["representation"] == "tfidf"
    assert prov["label_config"]["scheme"] == "windowed"
    assert prov["label_config"]["window_days"] == 3
    assert prov["dataset"]["n_rows"] == 60
    assert prov["run_at"]
    # code_revision is best-effort (None outside a git checkout), but the
    # key must exist so consumers can rely on the schema.
    assert "code_revision" in prov


def test_provenance_block_reports_embedding_cache_coverage(tmp_path, monkeypatch):
    """FinBERT runs record how many unique texts hit the embedding cache."""
    from signal_lab.models.labels import LabelConfig
    from signal_lab.models import embeddings as emb_mod
    from signal_lab.models.walk_forward import build_provenance

    monkeypatch.setattr(emb_mod, "default_cache_dir", lambda: tmp_path)
    (tmp_path / "deadbeef.npy").write_bytes(b"x")  # one cached entry
    monkeypatch.setattr(emb_mod, "cache_key", lambda t: "deadbeef" if t == "hit" else "miss-" + t)

    df = pd.DataFrame({
        "published_at": pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC"),
        "text": ["hit", "miss-a", "miss-b"],
        "label": [1, 0, 0],
    })
    prov = build_provenance("finbert_ctx", LabelConfig().validated(), df,
                            log=lambda *a, **k: None)
    ec = prov["embedding_cache"]
    assert ec["unique_texts"] == 3
    assert ec["cached"] == 1
    assert ec["missing"] == 2
    assert "raise" in ec["missing_policy"]
