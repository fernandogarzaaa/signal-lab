"""Tests for the TabPFN zero-tuning benchmark arm.

Covers: proba output contract, loud failures (missing package,
single-class conditioning, feature budget), PCA fit-on-train-only, and
the purged walk-forward wiring including the point-in-time conditioning
assertion (TabPFN conditions only on each fold's past train block).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_lab.models import tabpfn_arm as ta
from signal_lab.models.labels import LabelConfig
from signal_lab.validation import splits


def _blobs(n: int = 200, d: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    return X, y


def test_fit_predict_proba_contract():
    X, y = _blobs()
    p = ta.fit_predict_tabpfn(X[:150], y[:150], X[150:])
    assert p.shape == (50,)
    assert np.all(np.isfinite(p)) and p.min() >= 0.0 and p.max() <= 1.0
    # Sanity: it learned the blob structure better than chance.
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y[150:], p) > 0.7


def test_single_class_conditioning_raises():
    X, _ = _blobs()
    with pytest.raises(ValueError, match="single-class"):
        ta.fit_predict_tabpfn(X[:100], np.zeros(100, dtype=int), X[100:120])


def test_feature_budget_raises():
    X, y = _blobs(d=120)
    with pytest.raises(ValueError, match="feature"):
        ta.fit_predict_tabpfn(X[:150], y[:150], X[150:])


def test_pca_fit_on_train_only():
    rng = np.random.default_rng(4)
    Xtr = rng.normal(size=(120, 60))
    Xte = rng.normal(loc=5.0, size=(30, 60))  # shifted test distribution
    Ztr, Zte, pca = ta.pca_features(Xtr, Xte, n_components=10)
    assert Ztr.shape == (120, 10) and Zte.shape == (30, 10)
    assert np.all(np.isfinite(Ztr)) and np.all(np.isfinite(Zte))
    # The PCA must not have seen the test distribution: fitting on all
    # rows would give a different transform of the test rows.
    from sklearn.decomposition import PCA

    pca_all = PCA(n_components=10, random_state=7).fit(np.vstack([Xtr, Xte]))
    assert not np.allclose(pca.transform(Xte), pca_all.transform(Xte))


def _synthetic_frame(n: int = 360, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-06", periods=n, freq="B")
    words = np.where(rng.random(n) > 0.5, "bullish rally gains", "bearish slump losses")
    abn = rng.normal(scale=0.02, size=n)
    df = pd.DataFrame(
        {
            "text": words,
            "published_at": dates + pd.to_timedelta(rng.integers(9, 16, n), unit="h"),
            "ticker": np.where(rng.random(n) > 0.5, "AAA", "BBB"),
            "t0": dates.date,
            "t1": (dates + pd.offsets.BDay(3)).date,
            "abn_ret": abn,
            "ctx_asof": (dates - pd.offsets.BDay(1)).date,
        }
    )
    df["pub_date"] = pd.to_datetime(df["published_at"], utc=True).dt.date
    return df


def test_walk_forward_runs_and_reports():
    df = _synthetic_frame()
    trading_days = np.sort(pd.to_datetime(df["t0"]).map(pd.Timestamp.toordinal).unique())
    res = ta.run_tabpfn_walk_forward(
        df,
        n_splits=3,
        trading_days=trading_days,
        label_cfg=LabelConfig(attention_enabled=False),
        min_train=40,
        log=lambda *a, **k: None,
    )
    assert res["model"] == "tabpfn_zero_tuning"
    assert len(res["folds"]) > 0
    for f in res["folds"]:
        assert 0.0 <= f["pr_auc"] <= 1.0
        assert f["pr_auc"] >= 0.0
        assert f["baseline_pr_auc"] > 0.0
        assert f["pca_components"] <= 50
    assert res["validation"]["dedupe"].startswith("first")


def test_conditioning_is_point_in_time(monkeypatch):
    """The TabPFN conditioning set for fold k must be past-only.

    Spy on fit_predict_tabpfn to capture per-fold train sizes, then
    check against the harness splits that every conditioning row's t0
    predates its fold's test start.
    """
    df = _synthetic_frame()
    trading_days = np.sort(pd.to_datetime(df["t0"]).map(pd.Timestamp.toordinal).unique())
    seen = []
    real = ta.fit_predict_tabpfn

    def spy(Xtr, ytr, Xte, device="cpu"):
        seen.append((Xtr.shape[0], Xte.shape[0]))
        return real(Xtr, ytr, Xte, device=device)

    monkeypatch.setattr(ta, "fit_predict_tabpfn", spy)
    res = ta.run_tabpfn_walk_forward(
        df,
        n_splits=3,
        trading_days=trading_days,
        label_cfg=LabelConfig(attention_enabled=False),
        min_train=40,
        log=lambda *a, **k: None,
    )
    assert len(seen) == len(res["folds"])
    # Cross-check sizes against the harness splits (dedupe-first shrinks
    # train to unique ticker-t0 groups).
    work = df.dropna(subset=["published_at", "t0", "t1", "abn_ret"]).reset_index(drop=True)
    cfg = splits.WalkForwardConfig(n_splits=3, horizon_days=3, embargo_days=5,
                                   purge=True, min_train=40, retrain_every=1)
    wf = splits.make_splits(work, cfg, trading_days)
    for (n_tr, n_te), sp, f in zip(seen, wf, res["folds"]):
        assert n_te == f["n_test"] == len(sp["test_idx"])
        assert n_tr == f["n_train"]
        t0_max = pd.to_datetime(work["t0"].iloc[sp["train_idx"]]).max()
        assert t0_max < pd.Timestamp(sp["test_start"]), (
            "conditioning set reached into the test range"
        )


def test_pca_drops_degenerate_output_components():
    """Rank-deficient input must not yield near-zero-variance PCA output.

    Regression test: degenerate output components (train std ~1e-110)
    underflow to 0 in float32 and trigger environment-dependent NaNs
    inside TabPFN's encoder (observed as flaky CI failures on some
    runner CPUs). pca_features must drop them.
    """
    rng = np.random.default_rng(11)
    base = rng.normal(size=(80, 5))
    proj = rng.normal(size=(5, 15))
    # 20 input features of rank 5 plus 1e-13 noise: PCA will emit
    # degenerate near-zero-variance trailing components.
    Xtr = np.hstack([base, base @ proj + 1e-13 * rng.normal(size=(80, 15))])
    Xte = rng.normal(size=(20, 20))
    Ztr, Zte, _ = ta.pca_features(Xtr, Xte, n_components=15)
    assert Ztr.shape[1] < 15, "degenerate components were not dropped"
    assert np.all(Ztr.std(axis=0) > 1e-12)
    assert np.all(np.isfinite(Ztr)) and np.all(np.isfinite(Zte))
