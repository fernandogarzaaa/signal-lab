"""Tests for the TAILQ SAV CAViaR implementation (synthetic, no network).

Covers: the SAV recursion math, recovery of known quantile dynamics on
synthetic data generated from a known SAV process, forecast scale
(sqrt(3) horizon factor), short-window and degenerate fallbacks, and
the per-t0 forecast entry point.
"""

import numpy as np
import pytest

from signal_lab.tailq import caviar as cv
from signal_lab.tailq import models as models_mod


def _synthetic_sav(n, b1, b2, b3, seed=7):
    """Generate returns from a known SAV CAViaR process at tau=0.05.

    f_t = b1 + b2*f_{t-1} + b3*|r_{t-1}|; r_t = f_t + eps_t where
    eps_t is exponential noise shifted so that P(eps_t < 0) = 0.05
    (i.e. the true conditional 5% quantile of r_t is exactly f_t).
    """
    rng = np.random.default_rng(seed)
    f = np.empty(n)
    r = np.empty(n)
    f[0] = b1 / (1.0 - b2)
    for t in range(n):
        eps = rng.exponential(1.0) - np.log(1.0 / 0.95)
        r[t] = f[t] + eps * 0.02
        if t + 1 < n:
            f[t + 1] = b1 + b2 * f[t] + b3 * abs(r[t])
    return r, f


def test_sav_forward_path_math():
    r = np.array([0.01, -0.02, 0.015, -0.005, 0.008])
    beta = np.array([-0.02, 0.9, 0.1])
    f0 = -0.025
    f = cv._forward_path(beta, r, f0)
    assert f[0] == f0
    for t in range(1, len(r)):
        assert f[t] == pytest.approx(beta[0] + beta[1] * f[t - 1]
                                     + beta[2] * abs(r[t - 1]))


def test_check_loss_is_pinball_sum():
    r = np.array([0.01, -0.03, -0.10])
    f = np.array([-0.02, -0.02, -0.02])
    assert cv._check_loss(r - f, 0.05) == pytest.approx(
        float(np.sum((r - f) * (0.05 - ((r - f) < 0).astype(float)))))


def test_fit_recovers_known_sav_parameters():
    # True process: b1=-0.01, b2=0.85, b3=0.25 at tau=0.05.
    r, f_true = _synthetic_sav(600, b1=-0.01, b2=0.85, b3=0.25, seed=7)
    fit = cv.fit_caviar_sav(r, tau=0.05, window=600, seed=7)
    assert fit.converged, fit.detail
    # Quantile estimation on 600 points is noisy; require the dynamics
    # parameters within loose tolerance and the in-sample path to
    # track the true conditional quantile.
    assert fit.b2 == pytest.approx(0.85, abs=0.15)
    assert fit.b3 == pytest.approx(0.25, abs=0.15)
    f_hat = cv._forward_path(np.array([fit.b1, fit.b2, fit.b3]), r,
                             float(np.quantile(r, 0.05)))
    corr = np.corrcoef(f_hat[50:], f_true[50:])[0, 1]
    assert corr > 0.7


def test_forecast_has_sqrt3_horizon_scale():
    r, _ = _synthetic_sav(400, b1=-0.01, b2=0.85, b3=0.25, seed=7)
    fit = cv.fit_caviar_sav(r, window=400, seed=7)
    assert fit.converged
    f_next = fit.b1 + fit.b2 * fit.f_last + fit.b3 * abs(r[-1])
    assert cv.forecast_caviar_3d(fit, r[-1]) == pytest.approx(
        f_next * np.sqrt(3.0))
    assert cv.forecast_caviar_3d(fit, r[-1]) < 0


def test_forecast_at_t0_falls_back_on_short_window():
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.01, 50)
    q, fell_back = cv.forecast_at_t0(r, t0_pos=49, window=252)
    assert fell_back is True
    want = models_mod.naive_3d_quantile(r[:50], window=252)
    assert q == pytest.approx(want)
    assert q < 0


def test_forecast_at_t0_uses_returns_through_t0_only():
    # Appending post-t0 returns must not change the forecast at t0.
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.01, 400)
    q1, _ = cv.forecast_at_t0(r, t0_pos=299, window=252)
    r2 = np.concatenate([r, rng.normal(0, 0.05, 50)])
    q2, _ = cv.forecast_at_t0(r2, t0_pos=299, window=252)
    assert q1 == pytest.approx(q2)


def test_fit_rejects_nonfinite_returns():
    r = np.full(300, 0.001)
    r[100] = np.nan
    fit = cv.fit_caviar_sav(r, window=300)
    assert fit.converged is False
    assert "non-finite" in fit.detail


def test_garch_hs_point_in_time_and_scale():
    from signal_lab.tailq import garch_hs as ghs
    rng = np.random.default_rng(9)
    r = rng.normal(0, 0.01, 400)
    q1, fb1 = ghs.forecast_garch_hs_3d(r, t0_pos=299)
    assert fb1 is False
    assert np.isfinite(q1) and q1 < 0
    r2 = np.concatenate([r, rng.normal(0, 0.05, 50)])
    q2, _ = ghs.forecast_garch_hs_3d(r2, t0_pos=299)
    assert q1 == pytest.approx(q2)


def test_garch_hs_falls_back_on_short_window():
    from signal_lab.tailq import garch_hs as ghs
    rng = np.random.default_rng(9)
    r = rng.normal(0, 0.01, 50)
    q, fell_back = ghs.forecast_garch_hs_3d(r, t0_pos=49)
    assert fell_back is True
    assert q == pytest.approx(models_mod.naive_3d_quantile(r, window=252))
