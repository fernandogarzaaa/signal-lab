"""Tests for TAILQ metrics: pinball math, coverage tests, paired-t (synthetic).

No network. Verifies: pinball is the quantile check function, Kupiec
passes at the nominal rate and fails under excess violations,
Christoffersen independence passes on iid hits and fails on clustered
hits, and the paired-t verdict math matches a hand computation.
"""

import numpy as np
import pandas as pd

from signal_lab.tailq import metrics as met


def test_pinball_is_check_function():
    y = np.array([0.01, -0.03, 0.0, -0.10])
    q = np.array([-0.02, -0.02, -0.02, -0.02])
    tau = 0.05
    got = met.pinball(y, q, tau=tau)
    diff = y - q
    want = diff * (tau - (diff < 0).astype(float))
    np.testing.assert_allclose(got, want)
    # Over-prediction of a loss (q too high) costs more per unit than
    # under-prediction, at tau=0.05.
    assert met.pinball([-0.10], [-0.02])[0] > met.pinball([0.0], [-0.02])[0]


def test_pinball_known_values():
    # y >= q: loss = (y - q) * tau; y < q: loss = (q - y) * (1 - tau).
    assert met.pinball([0.03], [-0.02], tau=0.05)[0] == 0.05 * 0.05
    assert met.pinball([-0.10], [-0.02], tau=0.05)[0] == 0.08 * 0.95


def test_pinball_rejects_nonfinite():
    import pytest
    with pytest.raises(ValueError):
        met.pinball([0.01, np.nan], [-0.02, -0.02])
    with pytest.raises(ValueError):
        met.pinball([0.01], [np.inf])


def test_mean_pinball_fold_level():
    y = np.array([-0.05, -0.01, 0.02])
    q = np.array([-0.03, -0.03, -0.03])
    assert met.mean_pinball(y, q) == float(np.mean(met.pinball(y, q)))


def test_paired_t_matches_hand_computation():
    d = np.array([0.149621, -0.013392, -0.028122, 0.030955, -0.081545])
    got = met.paired_t(d)
    assert got["mean"] == float(np.mean(d))
    assert got["se"] == float(np.std(d, ddof=1) / np.sqrt(5))
    assert got["df"] == 4
    # 95% t CI with t(0.975, 4) = 2.7764.
    assert got["ci_low"] < got["mean"] < got["ci_high"]
    half = 2.7764 * got["se"]
    assert abs((got["ci_high"] - got["ci_low"]) / 2 - half) < 1e-4


def test_kupiec_passes_at_nominal_rate():
    rng = np.random.default_rng(7)
    hits = (rng.random(2000) < 0.05).astype(int)
    res = met.kupiec_pof(hits, p0=0.05)
    assert res["pass"] is True
    assert res["n"] == 2000
    assert 0.0 < res["p"] <= 1.0


def test_kupiec_fails_on_excess_violations():
    rng = np.random.default_rng(7)
    hits = (rng.random(2000) < 0.20).astype(int)
    res = met.kupiec_pof(hits, p0=0.05)
    assert res["pass"] is False
    assert res["stat"] > 0


def test_christoffersen_passes_on_iid_hits():
    rng = np.random.default_rng(11)
    hits = (rng.random(3000) < 0.05).astype(int)
    res = met.christoffersen_independence(hits)
    assert res["pass"] is True


def test_christoffersen_fails_on_clustered_hits():
    rng = np.random.default_rng(11)
    hits = np.zeros(3000, dtype=int)
    # Plant violation clusters: violations arrive in runs of 5.
    starts = rng.choice(2900, size=60, replace=False)
    for s in starts:
        hits[s:s + 5] = 1
    res = met.christoffersen_independence(hits)
    assert res["pass"] is False
    assert res["counts"]["n11"] > 0


def test_conditional_coverage_combines():
    rng = np.random.default_rng(7)
    hits = (rng.random(2000) < 0.05).astype(int)
    cc = met.conditional_coverage(hits)
    uc = met.kupiec_pof(hits)
    ind = met.christoffersen_independence(hits)
    assert abs(cc["stat"] - (uc["stat"] + ind["stat"])) < 1e-9
    assert cc["pass"] is True


def test_hits_from_quantile_forecasts():
    # A forecast at the true quantile has ~5% violations on average.
    rng = np.random.default_rng(3)
    y = rng.normal(0, 0.02, 5000)
    q = np.full(5000, -0.02 * 1.6448536269514722)
    hits = (y < q).astype(int)
    res = met.kupiec_pof(hits)
    assert abs(res["rate"] - 0.05) < 0.01
    assert res["pass"] is True
