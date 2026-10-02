"""Tests for vol.metrics: QLIKE values, fail-loud guards, DM equivalence."""

import numpy as np
import pytest
from scipy import stats

from signal_lab.vol import metrics as m


def test_qlike_known_value():
    # ratio = 0.04/0.05 = 0.8; q = 0.8 - ln(0.8) - 1
    got = m.qlike(np.array([0.04]), np.array([0.05]))
    expected = 0.8 - np.log(0.8) - 1.0
    assert got[0] == pytest.approx(expected, rel=1e-12)
    assert got[0] == pytest.approx(0.02314355, rel=1e-6)


def test_qlike_perfect_forecast_is_zero():
    got = m.qlike(np.array([0.04, 0.09]), np.array([0.04, 0.09]))
    assert np.allclose(got, 0.0)


def test_qlike_asymmetric_penalty():
    # under-prediction of variance is penalized more than over-prediction
    under = m.qlike(np.array([0.04]), np.array([0.02]))[0]
    over = m.qlike(np.array([0.04]), np.array([0.08]))[0]
    assert under > over > 0


def test_qlike_fails_loud():
    for bad_true, bad_hat in [
        (np.array([0.04]), np.array([0.0])),
        (np.array([0.04]), np.array([-0.01])),
        (np.array([0.0]), np.array([0.05])),
        (np.array([np.nan]), np.array([0.05])),
        (np.array([np.inf]), np.array([0.05])),
    ]:
        with pytest.raises(ValueError):
            m.qlike(bad_true, bad_hat)


def test_diebold_mariano_equals_paired_t():
    d = np.array([0.10, -0.05, 0.20, 0.15, 0.05])
    dm = m.diebold_mariano(d)
    t_res = stats.ttest_1samp(d, 0.0)
    assert dm["statistic"] == pytest.approx(t_res.statistic, rel=1e-12)
    assert dm["p_value"] == pytest.approx(t_res.pvalue, rel=1e-12)
    assert dm["n"] == 5


def test_t_confidence_interval_known_values():
    d = np.array([0.10, -0.05, 0.20, 0.15, 0.05])
    ci = m.t_confidence_interval(d, alpha=0.05)
    assert ci["mean"] == pytest.approx(d.mean(), rel=1e-12)
    assert ci["df"] == 4
    tcrit = stats.t.ppf(0.975, 4)
    assert ci["tcrit"] == pytest.approx(tcrit, rel=1e-12)
    se = d.std(ddof=1) / np.sqrt(5)
    assert ci["lower"] == pytest.approx(d.mean() - tcrit * se, rel=1e-12)
    assert ci["upper"] == pytest.approx(d.mean() + tcrit * se, rel=1e-12)


def test_mse_variance_and_mae_vol():
    vt = np.array([0.04, 0.09])
    vh = np.array([0.05, 0.08])
    assert m.mse_variance(vt, vh) == pytest.approx(((0.01) ** 2 + (0.01) ** 2) / 2)
    assert m.mae_vol(np.sqrt(vt), np.sqrt(vh)) == pytest.approx(
        (abs(0.2 - np.sqrt(0.05)) + abs(0.3 - np.sqrt(0.08))) / 2)
