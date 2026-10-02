"""Tests for vol.garch: forecast math, MLE recovery, fallbacks, panel."""

import numpy as np
import pandas as pd
import pytest

from signal_lab.vol import garch as garch_mod
from signal_lab.vol.garch import GarchFit


def _hand_forecast(omega, alpha, beta, sigma2_last, r_last, horizon=5):
    """Independent longhand of the pre-registered term structure."""
    sbar2 = omega / (1.0 - alpha - beta)
    s2_1 = omega + alpha * r_last ** 2 + beta * sigma2_last
    total = 0.0
    for h in range(1, horizon + 1):
        total += sbar2 + (alpha + beta) ** (h - 1) * (s2_1 - sbar2)
    return total ** 0.5


def test_forecast_math_vs_hand_computation():
    fit = GarchFit(omega=0.0001, alpha=0.08, beta=0.88,
                   sigma2_last=0.0004, n=252, converged=True, detail="ok")
    got = garch_mod.forecast_garch11(fit, r_last=0.01)
    expected = _hand_forecast(0.0001, 0.08, 0.88, 0.0004, 0.01)
    assert got == pytest.approx(expected, rel=1e-12)
    # hand value: sbar2=0.0025, s2_1=0.00046, total=0.003084 -> ~0.0555
    assert got == pytest.approx(0.05553, rel=1e-3)


def test_forecast_rejects_failed_fit():
    fit = GarchFit(omega=np.nan, alpha=np.nan, beta=np.nan,
                   sigma2_last=np.nan, n=10, converged=False,
                   detail="short-window:10<252")
    with pytest.raises(ValueError):
        garch_mod.forecast_garch11(fit, r_last=0.01)


def _simulate_garch11(n, omega, alpha, beta, seed=7):
    rng = np.random.default_rng(seed)
    sbar2 = omega / (1 - alpha - beta)
    s2 = np.empty(n)
    r = np.empty(n)
    s2[0] = sbar2
    for t in range(n):
        r[t] = rng.normal(0, np.sqrt(s2[t]))
        if t + 1 < n:
            s2[t + 1] = omega + alpha * r[t] ** 2 + beta * s2[t]
    return r


def test_mle_recovers_garch_params():
    # With the frozen 252-observation window, GARCH(1,1) MLE has large
    # sampling variation, so exact recovery is not asserted. What IS
    # asserted: convergence, stationarity, positive variance, and that the
    # fitted likelihood beats the constant-variance model on the same
    # window (the optimizer genuinely maximizes).
    r = _simulate_garch11(3000, omega=2e-6, alpha=0.08, beta=0.88)
    fit = garch_mod.fit_garch11(r)
    assert fit.converged, fit.detail
    assert fit.alpha + fit.beta < 1.0
    assert fit.sigma2_last > 0
    rw = r[-252:]
    r2 = rw * rw
    var0 = float(np.var(rw, ddof=1))
    nll_fit, _ = garch_mod._negloglik_and_grad(
        np.array([fit.omega, fit.alpha, fit.beta]), rw, r2, var0)
    nll_const, _ = garch_mod._negloglik_and_grad(
        np.array([var0 * 0.999, 1e-8, 1e-8]), rw, r2, var0)
    assert nll_fit < nll_const


def test_mle_param_recovery_long_window():
    # With more data the MLE lands in a plausible neighborhood of truth.
    r = _simulate_garch11(3000, omega=2e-6, alpha=0.08, beta=0.88)
    fit = garch_mod.fit_garch11(r, window=1000)
    assert fit.converged, fit.detail
    assert 0.03 < fit.alpha < 0.20
    assert 0.60 < fit.beta < 0.97
    assert fit.alpha + fit.beta < 1.0


def test_short_window_falls_back_to_naive():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, 100)
    fit = garch_mod.fit_garch11(r)
    assert not fit.converged
    assert "short-window" in fit.detail
    fc, fell_back = garch_mod.forecast_at_t0(r, t0_pos=99)
    assert fell_back
    expected_naive = float(np.std(r[95:100], ddof=1))
    assert fc == pytest.approx(expected_naive, rel=1e-12)


def test_degenerate_variance_falls_back():
    r = np.zeros(300)
    fit = garch_mod.fit_garch11(r)
    assert not fit.converged
    assert fit.detail == "degenerate-variance"


def test_naive_forecast_math():
    r = np.array([0.01, -0.02, 0.015, -0.005, 0.02, 0.008])
    got = garch_mod.naive_forecast(r, t0_pos=5)
    assert got == pytest.approx(float(np.std(r[1:6], ddof=1)), rel=1e-12)


def test_forecast_panel_tiny():
    dates = pd.bdate_range("2022-01-03", periods=400)
    rng = np.random.default_rng(9)
    rets = {}
    for t in ("AAA", "BBB"):
        rr = rng.normal(0.0005, 0.01, 400)
        rets[t] = pd.Series(rr, index=dates)
    rows = pd.DataFrame({
        "ticker": ["AAA"] * 3 + ["BBB"] * 3,
        "t0": list(dates[300:303].date) + list(dates[310:313].date),
    })
    panel = garch_mod.forecast_panel(rows, rets, n_jobs=1, log=lambda *a: None)
    assert len(panel) == 6
    assert (panel["garch_vol"] > 0).all()
    # determinism: same input -> same output
    panel2 = garch_mod.forecast_panel(rows, rets, n_jobs=1, log=lambda *a: None)
    pd.testing.assert_frame_equal(panel, panel2)
