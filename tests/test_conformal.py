"""Tests for split-conformal abstention (signal_lab.models.conformal).

Covers: the finite-sample coverage guarantee on synthetic data, the
abstain-on-empty-or-non-singleton rule, the rolling calibration window
(recency, not full history), and integration with the purged
walk-forward harness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from signal_lab.models import conformal as cf
from signal_lab.models.labels import LabelConfig


def _separable(n: int = 1200, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


def test_coverage_guarantee_on_synthetic():
    X, y = _separable()
    tr, ca, te = X[:600], X[600:1000], X[1000:]
    ytr, yca, yte = y[:600], y[600:1000], y[1000:]
    clf = LogisticRegression(max_iter=1000).fit(tr, ytr)
    q = cf.conformal_quantile(cf.nonconformity_scores(yca, clf.predict_proba(ca)[:, 1]), 0.1)
    sets = cf.prediction_sets(clf.predict_proba(te)[:, 1], q)
    cov = cf.empirical_coverage(yte, sets)
    # Finite-sample guarantee: coverage >= 0.9 up to binomial noise.
    assert cov >= 0.85, f"coverage {cov:.3f} below 0.85 at alpha=0.1"


def test_nonconformity_scores_definition():
    y = np.array([1, 0, 1])
    p = np.array([0.8, 0.7, 0.2])
    s = cf.nonconformity_scores(y, p)
    # 1 - p(true class): [0.2, 1 - 0.3, 0.8] = [0.2, 0.7, 0.8]
    assert np.allclose(s, [0.2, 0.7, 0.8])


def test_quantile_uses_finite_sample_correction():
    scores = np.arange(1, 101, dtype=float)  # 1..100
    q = cf.conformal_quantile(scores, 0.1)
    # ceil(101 * 0.9) = 91 -> 91st order statistic = 91.0 (naive quantile would give 90.1)
    assert q == 91.0, f"expected 91.0, got {q}"


def test_abstain_rule_empty_singleton_nonsingleton():
    sets = [frozenset(), frozenset({1}), frozenset({0, 1}), frozenset({0})]
    mask = cf.abstain_mask(sets)
    assert mask.tolist() == [True, False, True, False]
    preds = cf.singleton_predictions(sets)
    assert preds.tolist() == [-1, 1, -1, 0]


def test_prediction_sets_confident_vs_uncertain():
    q = 0.6
    sets = cf.prediction_sets(np.array([0.99, 0.5, 0.01]), q)
    assert sets[0] == frozenset({1})   # confident positive -> singleton
    assert sets[2] == frozenset({0})   # confident negative -> singleton
    assert sets[1] == frozenset({0, 1})  # uncertain -> both labels -> abstain


def test_prediction_sets_empty_means_abstain():
    # q smaller than both nonconformity scores -> empty set -> abstain.
    sets = cf.prediction_sets(np.array([0.5]), 0.2)
    assert sets[0] == frozenset()
    assert cf.abstain_mask(sets).tolist() == [True]


def test_rolling_window_uses_recent_calibration_only():
    rng = np.random.default_rng(3)
    # Early calibration: model badly wrong (scores near 1). Recent: good (scores near 0.05).
    y = np.ones(200, dtype=int)
    p = np.concatenate([np.full(150, 0.05), np.full(50, 0.95)])
    q_full = cf.fit_rolling_quantile(y, p, alpha=0.1, cal_window=200)
    q_recent = cf.fit_rolling_quantile(y, p, alpha=0.1, cal_window=50)
    assert q_recent < q_full, f"rolling q={q_recent} should be < full q={q_full}"
    assert q_recent <= 0.06


def test_rolling_window_raises_when_too_small():
    y = np.ones(10, dtype=int)
    p = np.full(10, 0.9)
    try:
        cf.fit_rolling_quantile(y, p, alpha=0.1, cal_window=50)
    except ValueError as exc:
        assert "cal_window" in str(exc)
    else:
        raise AssertionError("expected ValueError for short calibration block")


def _synthetic_frame(n: int = 600, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-06", periods=n, freq="B")
    words = np.where(rng.random(n) > 0.5, "bullish rally gains", "bearish slump losses")
    abn = rng.normal(scale=0.02, size=n) + np.where(
        rng.random(n) > 0.5, 0.01, -0.01
    )
    df = pd.DataFrame(
        {
            "text": words,
            "published_at": dates + pd.to_timedelta(rng.integers(9, 16, n), unit="h"),
            "ticker": "SYN",
            "t0": dates.date,
            "t1": (dates + pd.offsets.BDay(3)).date,
            "abn_ret": abn,
            "ctx_asof": (dates - pd.offsets.BDay(1)).date,
        }
    )
    df["pub_date"] = pd.to_datetime(df["published_at"], utc=True).dt.date
    return df


def test_walk_forward_integration_reports_abstention():
    df = _synthetic_frame()
    trading_days = np.sort(
        pd.to_datetime(df["t0"]).map(pd.Timestamp.toordinal).unique()
    )
    res = cf.run_conformal_walk_forward(
        df,
        n_splits=3,
        trading_days=trading_days,
        label_cfg=LabelConfig(attention_enabled=False),
        alpha=0.2,
        cal_frac=0.25,
        cal_window=30,
        min_train=40,
        log=lambda *a, **k: None,
    )
    assert res["method"].startswith("split-conformal")
    assert len(res["folds"]) > 0, "expected at least one completed fold"
    for f in res["folds"]:
        assert 0.0 <= f["abstention_rate"] <= 1.0
        assert 0.0 <= f["empirical_coverage"] <= 1.0
        assert f["n_cal"] >= 30
        assert f["q_hat"] >= 0.0
    assert res["skipped_folds"] == [] or all(
        "reason" in s for s in res["skipped_folds"]
    )


def test_walk_forward_integration_never_touches_confirmation():
    from datetime import date

    from signal_lab.validation.periods import ConfirmationLeakError

    df = _synthetic_frame()
    trading_days = np.sort(
        pd.to_datetime(df["t0"]).map(pd.Timestamp.toordinal).unique()
    )
    # A dev_end inside the frozen confirmation period must fail loudly in
    # make_splits (check_no_confirmation), never silently train on it.
    try:
        cf.run_conformal_walk_forward(
            df,
            n_splits=3,
            trading_days=trading_days,
            label_cfg=LabelConfig(attention_enabled=False),
            cal_window=30,
            min_train=40,
            dev_end=date(2026, 9, 15),
            log=lambda *a, **k: None,
        )
    except ConfirmationLeakError:
        pass
    else:
        raise AssertionError("expected ConfirmationLeakError for confirmation dev_end")
