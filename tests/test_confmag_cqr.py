"""Tests for confmag.cqr: CQR construction math, selection rule, temporal split."""

import numpy as np
import pandas as pd
import pytest

from signal_lab.confmag import cqr as cqr_mod
from signal_lab.models.conformal import conformal_quantile


class _FakeModel:
    """Deterministic stand-in with fixed quantile predictions."""

    def __init__(self, vals):
        self._vals = np.asarray(vals, dtype=float)

    def predict(self, X):
        n = len(X)
        if len(self._vals) == n:
            return self._vals.copy()
        # Broadcast a constant when shapes differ (test-only).
        return np.full(n, float(self._vals[0]))


def test_conformalize_math():
    # y = [1, 2, 3, 4], q_lo = y + 0.2 (too high), q_hi = y + 1.0.
    # scores E_i = max(0.2, -1.0) = 0.2 for all rows.
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    X = pd.DataFrame({"a": [0.0] * 4})
    m_lo = _FakeModel(y.to_numpy() + 0.2)
    m_hi = _FakeModel(y.to_numpy() + 1.0)
    out = cqr_mod.conformalize(m_lo, m_hi, X, y, alpha=0.2)
    expected_q = conformal_quantile(np.full(4, 0.2), 0.2)
    assert out["q_hat"] == pytest.approx(expected_q)
    # widths = (q_hi - q_lo) + 2*q_hat = 0.8 + 2*0.2 = 1.2;
    # threshold = 25th pct of constant widths.
    assert out["threshold"] == pytest.approx(0.8 + 2.0 * expected_q)
    assert out["n_cal"] == 4


def test_selection_rule():
    y = pd.Series(np.arange(10, dtype=float))
    X = pd.DataFrame({"a": np.arange(10, dtype=float)})
    # Varying widths: q_hi - q_lo = 1 + x/10.
    m_lo = _FakeModel(np.zeros(10))
    m_hi = _FakeModel(1.0 + np.arange(10) / 10.0)
    out = cqr_mod.conformalize(m_lo, m_hi, X, y, alpha=0.2)
    # y in [0..9], intervals [0 - q, 1+x/10 + q]: scores = max(-y, y - (1+x/10)).
    # Just check the selection machinery end to end.
    iv = cqr_mod.predict_intervals(m_lo, m_hi, X, out["q_hat"])
    widths = iv["widths"]
    assert len(widths) == 10
    sel = widths <= out["threshold"]
    # Threshold is the 25th percentile -> ~25% selected (discrete).
    assert 1 <= sel.sum() <= 4
    # Selected rows are the tightest intervals.
    assert widths[sel].max() <= widths[~sel].min()


def test_temporal_proper_cal_split():
    t0 = pd.Series(pd.bdate_range("2022-01-03", periods=100))
    proper, cal = cqr_mod.temporal_proper_cal(t0, cal_frac=0.2)
    assert len(cal) == 20
    assert len(proper) == 80
    # Calibration is the most recent 20%.
    assert t0.iloc[cal].min() > t0.iloc[proper].max()
    assert set(proper) | set(cal) == set(range(100))


def test_predict_intervals_math():
    X = pd.DataFrame({"a": [1.0, 2.0]})
    m_lo = _FakeModel([0.5, 1.5])
    m_hi = _FakeModel([2.5, 3.5])
    out = cqr_mod.predict_intervals(m_lo, m_hi, X, q_hat=0.25)
    assert np.allclose(out["lo"], [0.25, 1.25])
    assert np.allclose(out["hi"], [2.75, 3.75])
    assert np.allclose(out["widths"], [2.5, 2.5])


def test_empirical_coverage():
    y = np.array([1.0, 5.0, 3.0])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([2.0, 4.0, 4.0])
    assert cqr_mod.empirical_coverage(y, lo, hi) == pytest.approx(2 / 3)


def test_point_forecast_floored_at_zero():
    class _Neg:
        def predict(self, X):
            return np.full(len(X), -3.0)

    X = pd.DataFrame({"a": [1.0]})
    assert cqr_mod.predict_point(_Neg(), X)[0] == 0.0


def test_quantile_params_frozen():
    p_lo = cqr_mod._quantile_params(0.1)
    assert p_lo["objective"] == "quantile"
    assert p_lo["alpha"] == 0.1
    assert p_lo["n_estimators"] == 300
    assert p_lo["random_state"] == 7
    assert cqr_mod.TAU_LO == 0.1 and cqr_mod.TAU_HI == 0.9
    assert cqr_mod.ALPHA == 0.2 and cqr_mod.CAL_FRAC == 0.2
    assert cqr_mod.SELECT_Q == 0.25
