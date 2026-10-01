"""Tests for the gate-2 FinBERT arm in signal_lab.benchmark.phase3.

Uses a synthetic FinBERT cache (fake 768-dim vectors keyed exactly like
the real cache) so no torch/transformers/network is needed. The real
dataset exercises the full fold loop; the cache-only contract (missing
embedding raises loudly) is tested directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_lab.benchmark import phase3
from signal_lab.models.build_and_train import build_dataset, load_trading_calendar_days
from signal_lab.models.embeddings import (
    MissingEmbeddingError,
    _cache_stem,
    cache_key,
)


def _fake_cache(texts, tmp_path) -> str:
    """Populate a directory with synthetic 768-dim embeddings for `texts`."""
    rng = np.random.default_rng(7)
    cdir = tmp_path / "finbert"
    cdir.mkdir()
    for t in dict.fromkeys(str(x) for x in texts):  # unique, order kept
        vec = rng.normal(0, 1, 768).astype(np.float32)
        np.save(cdir / f"{_cache_stem(cache_key(t))}.npy", vec)
    return str(cdir)


@pytest.fixture(scope="module")
def dev_frame():
    df, _ = build_dataset(log=lambda *a: None)
    trading_days = load_trading_calendar_days(log=lambda *a: None)
    return df, trading_days


def test_build_matrices_finbert_shapes_and_scaling(dev_frame, tmp_path):
    df, _ = dev_frame
    cdir = _fake_cache(df["text"], tmp_path)
    sub = df.iloc[:60]
    mtr, mte, _ = phase3._build_matrices(
        sub["text"].iloc[:40], sub["text"].iloc[40:],
        sub[phase3.PRICE_COLS].iloc[:40], sub[phase3.PRICE_COLS].iloc[40:],
        finbert_cache_dir=cdir,
    )
    n_price = len(phase3.PRICE_COLS)
    assert mtr["Fb"].shape == (40, 768)
    assert mte["Fb"].shape == (20, 768)
    assert mtr["Fc"].shape == (40, 768 + n_price)
    assert mte["Fc"].shape == (20, 768 + n_price)
    # StandardScaler was fit on train: train Fb columns ~ zero mean, unit var
    fb = mtr["Fb"].toarray()
    assert abs(fb.mean()) < 0.05
    assert 0.9 < fb.std() < 1.1


def test_build_matrices_missing_embedding_raises(dev_frame, tmp_path):
    df, _ = dev_frame
    cdir = tmp_path / "empty_cache"
    cdir.mkdir()
    sub = df.iloc[:10]
    with pytest.raises(MissingEmbeddingError):
        phase3._build_matrices(
            sub["text"].iloc[:6], sub["text"].iloc[6:],
            sub[phase3.PRICE_COLS].iloc[:6], sub[phase3.PRICE_COLS].iloc[6:],
            finbert_cache_dir=str(cdir),
        )


def test_gate_verdict_primary_comparison():
    paired = {
        "logreg_balanced": {
            "C-A/pr_auc": {"mean_diff": -0.01, "ci95": [-0.03, 0.01],
                           "t_pvalue": 0.5, "wilcoxon_pvalue": 0.6},
            "Fc-A/pr_auc": {"mean_diff": 0.04, "ci95": [0.01, 0.07],
                            "t_pvalue": 0.02, "wilcoxon_pvalue": 0.03},
        }
    }
    g1 = phase3.gate_verdict(paired)  # default C-A
    assert g1["verdict"] == "NO-GO"
    assert g1["primary_comparison"] == "C-A"
    g2 = phase3.gate_verdict(paired, primary_comparison="Fc-A")
    assert g2["verdict"] == "GO"
    assert g2["primary_comparison"] == "Fc-A"
    assert g2["primary_mean_diff"] == pytest.approx(0.04)


def test_config_rejects_bad_primary_comparison():
    with pytest.raises(ValueError, match="primary_comparison"):
        phase3.BenchmarkConfig(primary_comparison="Z-A").validated()
    with pytest.raises(ValueError, match="primary_comparison"):
        phase3.BenchmarkConfig(primary_comparison="nope").validated()
    cfg = phase3.BenchmarkConfig(
        primary_comparison=phase3.PRIMARY_COMPARISON_GATE2).validated()
    assert cfg.primary_comparison == "Fc-A"


def test_run_benchmark_finbert_arm_end_to_end(dev_frame, tmp_path):
    df, trading_days = dev_frame
    cdir = _fake_cache(df["text"], tmp_path)
    cfg = phase3.BenchmarkConfig(
        calibrate=False, abstain=False,
        primary_comparison=phase3.PRIMARY_COMPARISON_GATE2,
        model_names=("logreg_balanced",),
    )
    res = phase3.run_benchmark(df, trading_days, config=cfg,
                               finbert_cache_dir=cdir, log=lambda *a: None)
    assert set(res["aggregate"]) >= {"A", "B", "C", "Fb", "Fc"}
    p = res["paired"]["logreg_balanced"]
    assert "Fc-A/pr_auc" in p and "Fc-Fb/pr_auc" in p and "C-A/pr_auc" in p
    assert res["gate"]["primary_comparison"] == "Fc-A"
    assert res["gate"]["verdict"] in ("GO", "NO-GO")
    # every fold has metrics for every variant
    for f in res["folds"]:
        assert set(f["metrics"]) >= {"A", "B", "C", "Fb", "Fc"}
