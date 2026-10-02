"""TAILQ pipeline smoke test: walk-forward loop on synthetic data (no network).

Exercises the real splitter (WalkForwardConfig horizon=3, embargo=3,
seed=7, purged expanding walk-forward), fits the frozen challenger per
fold, scores pinball for challenger vs naive, and asserts the verdict
math runs end to end. CAViaR/GARCH arms are covered in
test_tailq_caviar.py; this test covers the orchestration.
"""

import numpy as np
import pandas as pd

from signal_lab.tailq import features as feat_mod
from signal_lab.tailq import metrics as met
from signal_lab.tailq import models as models_mod
from signal_lab.validation.splits import WalkForwardConfig, make_splits


def _synthetic_frame(n_events=400, seed=7):
    rng = np.random.default_rng(seed)
    t0s = pd.bdate_range("2023-01-02", periods=n_events)
    rows = []
    for i, d in enumerate(t0s):
        rows.append({
            "ticker": "AAA", "t0": d.date(),
            "t1": (d + pd.offsets.BDay(3)).date(),
            "published_at": pd.Timestamp(d, tz="UTC"),
            "target": float(rng.normal(0, 0.02)),
            **{c: float(v) for c, v in zip(
                feat_mod.FEATURE_COLUMNS, rng.normal(0, 1, 15))},
        })
    return pd.DataFrame(rows)


def test_walk_forward_smoke():
    feat = _synthetic_frame()
    cal = pd.bdate_range("2022-01-01", periods=400)
    trading_days = np.sort(np.array([d.date().toordinal() for d in cal]))
    cfg = WalkForwardConfig(n_splits=5, window="expanding",
                            horizon_days=3, embargo_days=3,
                            min_train=50, seed=7).validated()
    splits = make_splits(feat, cfg, trading_days)
    assert len(splits) == 5
    X = feat[feat_mod.FEATURE_COLUMNS].to_numpy(dtype=float)
    y = feat["target"].to_numpy(dtype=float)
    ds = []
    for s in splits:
        tri, tei = s["train_idx"], s["test_idx"]
        assert len(tri) >= 50
        assert len(tei) > 0
        # Purge invariant: no train row's [t0, t1] overlaps this fold's
        # test range.
        tr = feat.iloc[tri]
        assert (pd.to_datetime(tr["t1"]).dt.date
                < pd.Timestamp(s["test_start"]).date()).all()
        booster = models_mod.fit_lgbm_quantile(X[tri], y[tri])
        q = models_mod.predict_lgbm_quantile(booster, X[tei])
        assert np.all(np.isfinite(q))
        q_naive = np.full_like(q, np.quantile(y[tri], 0.05) * np.sqrt(3.0))
        pin_ch = met.mean_pinball(y[tei], q)
        pin_nv = met.mean_pinball(y[tei], q_naive)
        assert np.isfinite(pin_ch) and np.isfinite(pin_nv)
        ds.append(pin_nv - pin_ch)
    pt = met.paired_t(np.array(ds))
    assert set(pt) >= {"mean", "se", "ci_low", "ci_high", "stat", "p"}
    assert np.isfinite(pt["mean"])


def test_folds_are_temporally_ordered():
    feat = _synthetic_frame()
    cal = pd.bdate_range("2022-01-01", periods=400)
    trading_days = np.sort(np.array([d.date().toordinal() for d in cal]))
    cfg = WalkForwardConfig(n_splits=5, window="expanding",
                            horizon_days=3, embargo_days=3,
                            min_train=50, seed=7).validated()
    splits = make_splits(feat, cfg, trading_days)
    starts = [s["test_start"] for s in splits]
    assert starts == sorted(starts)
    for s in splits:
        assert s["test_start"] < s["test_end"]
