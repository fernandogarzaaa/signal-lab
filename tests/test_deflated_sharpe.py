"""Tests for stats.deflated_sharpe (APP-006).

The DSR implements Bailey & Lopez de Prado (2014) directly. Expected
values are computed through an independent path (stdlib statistics /
NormalDist / math.erf), not by re-calling the implementation, so a
transcription bug cannot hide behind the same code.
"""

import math
import statistics
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from signal_lab.stats import deflated_sharpe as dsr


def _ref_expected_sr0(trial_srs):
    n = len(trial_srs)
    var = statistics.variance(trial_srs)  # sample variance, ddof=1
    nd = NormalDist()
    term = ((1 - dsr.EULER_MASCHERONI) * nd.inv_cdf(1 - 1 / n)
            + dsr.EULER_MASCHERONI * nd.inv_cdf(1 - 1 / (n * math.e)))
    return math.sqrt(var) * term


def _ref_dsr(returns, trial_srs, freq=252):
    r = list(returns)
    mean = statistics.fmean(r)
    sd = statistics.stdev(r)
    sr_hat = mean / sd * math.sqrt(freq)
    sr_0 = _ref_expected_sr0(trial_srs)
    t = len(r)
    # population skew / Pearson kurtosis by direct definition
    m2 = statistics.fmean([(x - mean) ** 2 for x in r])
    m3 = statistics.fmean([(x - mean) ** 3 for x in r])
    m4 = statistics.fmean([(x - mean) ** 4 for x in r])
    skew = m3 / m2 ** 1.5
    kurt = m4 / m2 ** 2
    denom = 1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat ** 2
    stat = (sr_hat - sr_0) * math.sqrt(t - 1) / math.sqrt(denom)
    return NormalDist().cdf(stat)


RETURNS = [0.012, -0.004, 0.021, 0.007, -0.011, 0.018, 0.002,
           -0.007, 0.015, 0.009, -0.002, 0.011, 0.005, -0.009,
           0.014, 0.003, 0.008, -0.005, 0.017, 0.006]
TRIALS = [0.62, 0.41, 0.18, -0.12, 0.05, 0.33, -0.28, 0.11]


def test_expected_sharpe_null_zero_variance():
    # Identical trial Sharpes -> V = 0 -> SR_0 = 0 exactly.
    assert dsr.expected_sharpe_null([0.4, 0.4, 0.4, 0.4]) == 0.0


def test_expected_sharpe_null_matches_stdlib():
    assert dsr.expected_sharpe_null(TRIALS) == pytest.approx(
        _ref_expected_sr0(TRIALS), rel=1e-12
    )


def test_dsr_full_formula_matches_stdlib():
    assert dsr.deflated_sharpe_ratio(RETURNS, TRIALS) == pytest.approx(
        _ref_dsr(RETURNS, TRIALS), rel=1e-9
    )


def test_sharpe_ratio_known_value():
    r = [0.01, 0.02, -0.01]
    expected = statistics.fmean(r) / statistics.stdev(r) * math.sqrt(252)
    assert dsr.sharpe_ratio(r) == pytest.approx(expected, rel=1e-12)


def test_dsr_is_a_probability():
    rng = np.random.default_rng(3)
    for _ in range(10):
        rets = rng.normal(0.001, 0.02, 120)
        trials = list(rng.normal(0.2, 0.5, 12))
        v = dsr.deflated_sharpe_ratio(rets, trials)
        assert 0.0 <= v <= 1.0


def test_dsr_falls_as_trial_count_grows():
    # Same trial-SR distribution, more trials -> the expected maximum
    # under the null rises -> lower DSR. (Note: adding near-identical
    # trials instead shrinks the trial variance, which correctly lowers
    # the bar; the distribution must be held fixed for this property.)
    rng = np.random.default_rng(7)
    draws = list(rng.normal(0.3, 0.4, 1000))
    few, many = draws[:10], draws  # N = 10 vs N = 1000, same distribution
    r2 = np.random.default_rng(9)
    rets = list(r2.normal(0.000882, 0.02, 200))
    assert dsr.expected_sharpe_null(many) > dsr.expected_sharpe_null(few)
    d_few = dsr.deflated_sharpe_ratio(rets, few)
    d_many = dsr.deflated_sharpe_ratio(rets, many)
    assert d_few > d_many
    assert 0.0 <= d_many < d_few <= 1.0


def test_dsr_report_sorts_and_flags():
    rng = np.random.default_rng(5)
    variants = {
        "winner": list(rng.normal(0.004, 0.02, 200)),
        "flat": list(rng.normal(0.0, 0.02, 200)),
        "loser": list(rng.normal(-0.004, 0.02, 200)),
    }
    rep = dsr.dsr_report(variants)
    assert list(rep.columns) == ["variant", "sharpe", "dsr", "likely_false_discovery"]
    assert rep["dsr"].is_monotonic_decreasing
    assert rep["likely_false_discovery"].dtype == bool
    # The flat variant's Sharpe is ~0, so its DSR must sit below the
    # discovery threshold: selection bias is visible, not hidden.
    flat_dsr = rep.loc[rep["variant"] == "flat", "dsr"].iloc[0]
    assert flat_dsr < 0.95


def test_dsr_report_needs_all_tried_variants():
    with pytest.raises(ValueError, match="at least 2"):
        dsr.dsr_report({"only": [0.01, -0.02, 0.03]})


def test_errors_fail_loud():
    with pytest.raises(ValueError, match="at least 2 return"):
        dsr.sharpe_ratio([0.01])
    with pytest.raises(ValueError, match="NaN or inf"):
        dsr.sharpe_ratio([0.01, float("nan"), 0.02])
    with pytest.raises(ValueError, match="zero variance"):
        dsr.sharpe_ratio([0.01, 0.01, 0.01])
    with pytest.raises(ValueError, match="at least 2 tried"):
        dsr.expected_sharpe_null([0.5])
    with pytest.raises(ValueError, match="NaN or inf"):
        dsr.expected_sharpe_null([0.5, float("inf")])


def test_dsr_report_returns_dataframe():
    rep = dsr.dsr_report({"a": RETURNS, "b": [-x for x in RETURNS]})
    assert isinstance(rep, pd.DataFrame)
    assert len(rep) == 2
    assert (rep["dsr"] >= 0).all() and (rep["dsr"] <= 1).all()
