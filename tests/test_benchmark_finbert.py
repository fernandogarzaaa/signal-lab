"""Tests for the gate-2 FinBERT arm in signal_lab.benchmark.phase3.

Fully synthetic: no DuckDB, no network, no torch. The benchmark's
FinBERT arm is cache-only by design, so tests populate a throwaway
cache keyed exactly like the real one.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from signal_lab.benchmark import phase3
from signal_lab.benchmark.phase3 import (
    PRICE_COLS,
    BenchmarkConfig,
    gate_verdict,
    run_benchmark,
)
from signal_lab.models.embeddings import (
    MissingEmbeddingError,
    _cache_stem,
    cache_key,
)
from signal_lab.models.labels import LabelConfig


def _synthetic_df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    tickers = ["AAA", "BBB", "CCC"]
    base = date(2025, 2, 3)  # a Monday; all rows are development period
    words = ["market", "stock", "price", "trade", "news", "report",
             "earnings", "growth"]
    rows = []
    for i in range(n):
        d = base + timedelta(days=i // 3)
        abn = float(rng.normal(0, 0.03))
        rows.append({
            "ticker": tickers[i % 3],
            "pub_date": d,
            "published_at": pd.Timestamp(d) + pd.Timedelta(hours=8),
            "ctx_asof": pd.Timestamp(d) - pd.Timedelta(days=1),
            "t0": pd.Timestamp(d),
            "t1": pd.Timestamp(d) + pd.Timedelta(days=3),
            "abn_ret": abn,
            "excess_3d": abn,
            "direction_3d": 1.0 if abn > 0 else 0.0,
            "text": " ".join(rng.choice(words, size=12)),
        })
    df = pd.DataFrame(rows)
    for c in PRICE_COLS:
        df[c] = rng.normal(0, 1, n)
    return df


def _trading_days(df):
    days = pd.to_datetime(df["t0"]).dt.date.apply(lambda d: d.toordinal())
    return np.sort(days.unique().astype(np.int64))


def _no_attention():
    return LabelConfig(attention_enabled=False).validated()


def _fb_cache(texts, tmp_path):
    """Throwaway FinBERT cache with synthetic 768-dim vectors for `texts`."""
    rng = np.random.default_rng(7)
    cdir = tmp_path / "finbert"
    cdir.mkdir(exist_ok=True)
    for t in dict.fromkeys(str(x) for x in texts):
        vec = rng.normal(0, 1, 768).astype(np.float32)
        np.save(cdir / f"{_cache_stem(cache_key(t))}.npy", vec)
    return str(cdir)


def test_build_matrices_finbert_shapes_and_scaling(tmp_path):
    df = _synthetic_df()
    cdir = _fb_cache(df["text"], tmp_path)
    sub = df.iloc[:60]
    mtr, mte, _ = phase3._build_matrices(
        sub["text"].iloc[:40], sub["text"].iloc[40:],
        sub[PRICE_COLS].iloc[:40], sub[PRICE_COLS].iloc[40:],
        finbert_cache_dir=cdir,
    )
    n_price = len(PRICE_COLS)
    assert mtr["Fb"].shape == (40, 768)
    assert mte["Fb"].shape == (20, 768)
    assert mtr["Fc"].shape == (40, 768 + n_price)
    assert mte["Fc"].shape == (20, 768 + n_price)
    # StandardScaler was fit on train: train Fb columns ~ zero mean, unit var
    fb = mtr["Fb"].toarray()
    assert abs(fb.mean()) < 0.05
    assert 0.9 < fb.std() < 1.1


def test_build_matrices_missing_embedding_raises(tmp_path):
    df = _synthetic_df()
    cdir = tmp_path / "empty_cache"
    cdir.mkdir()
    sub = df.iloc[:10]
    with pytest.raises(MissingEmbeddingError):
        phase3._build_matrices(
            sub["text"].iloc[:6], sub["text"].iloc[6:],
            sub[PRICE_COLS].iloc[:6], sub[PRICE_COLS].iloc[6:],
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
    g1 = gate_verdict(paired)  # default C-A preserves gate-1 behavior
    assert g1["verdict"] == "NO-GO"
    assert g1["primary_comparison"] == "C-A"
    g2 = gate_verdict(paired, primary_comparison="Fc-A")
    assert g2["verdict"] == "GO"
    assert g2["primary_comparison"] == "Fc-A"
    assert g2["primary_mean_diff"] == pytest.approx(0.04)


def test_config_rejects_bad_primary_comparison():
    with pytest.raises(ValueError, match="primary_comparison"):
        BenchmarkConfig(primary_comparison="Z-A").validated()
    with pytest.raises(ValueError, match="primary_comparison"):
        BenchmarkConfig(primary_comparison="nope").validated()
    cfg = BenchmarkConfig(
        primary_comparison=phase3.PRIMARY_COMPARISON_GATE2).validated()
    assert cfg.primary_comparison == "Fc-A"


def test_run_benchmark_finbert_arm_end_to_end(tmp_path):
    df = _synthetic_df()
    cdir = _fb_cache(df["text"], tmp_path)
    cfg = BenchmarkConfig(
        n_splits=2, min_train=20, calibrate=False, abstain=False,
        primary_comparison=phase3.PRIMARY_COMPARISON_GATE2,
        model_names=("logreg_balanced",),
    ).validated()
    res = run_benchmark(df, _trading_days(df), label_cfg=_no_attention(),
                        config=cfg, finbert_cache_dir=cdir,
                        log=lambda *a: None)
    assert set(res["aggregate"]) >= {"A", "B", "C", "Fb", "Fc"}
    p = res["paired"]["logreg_balanced"]
    assert "Fc-A/pr_auc" in p and "Fc-Fb/pr_auc" in p and "C-A/pr_auc" in p
    assert res["gate"]["primary_comparison"] == "Fc-A"
    assert res["gate"]["verdict"] in ("GO", "NO-GO")
    for f in res["folds"]:
        assert set(f["metrics"]) >= {"A", "B", "C", "Fb", "Fc"}
