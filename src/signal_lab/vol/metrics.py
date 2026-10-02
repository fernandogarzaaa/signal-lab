"""VOL metrics: QLIKE (primary), MSE-on-variance, MAE-on-vol, Diebold-Mariano.

QLIKE (Patton 2011), consistent for ranking volatility forecasts and
robust to noise in the realized proxy; lower is better:
    QLIKE(sigma^2, sigma_hat^2) = sigma^2/sigma_hat^2 - ln(sigma^2/sigma_hat^2) - 1.

Diebold-Mariano note (documented equivalence): the campaign runs DM on
FOLD-LEVEL mean QLIKE loss differentials d_1..d_K (K = 5 folds). The DM
statistic is dbar / sqrt(LRV / K) where LRV is the long-run variance of
the differential series. With one observation per fold, the folds are
non-overlapping blocks, so the HAC long-run variance estimator collapses
to the ordinary sample variance of the fold means, and the DM statistic
is algebraically identical to the paired t-statistic
dbar / (s_d / sqrt(K)). We report the paired-t form directly.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _positive(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN or inf; refusing to score")
    if np.any(x <= 0):
        raise ValueError(
            f"{name} contains non-positive values; QLIKE needs strictly "
            "positive variances"
        )
    return x


def qlike(var_true, var_hat) -> np.ndarray:
    """Elementwise QLIKE on variance. Lower is better. Fail loud on bad input."""
    vt = _positive(var_true, "var_true")
    vh = _positive(var_hat, "var_hat")
    if vt.shape != vh.shape:
        raise ValueError(f"shape mismatch: {vt.shape} vs {vh.shape}")
    ratio = vt / vh
    return ratio - np.log(ratio) - 1.0


def mean_qlike(var_true, var_hat) -> float:
    """Mean QLIKE over a set of rows (e.g. one walk-forward fold)."""
    return float(np.mean(qlike(var_true, var_hat)))


def mse_variance(var_true, var_hat) -> float:
    """Mean squared error on variance (secondary)."""
    vt = _positive(var_true, "var_true")
    vh = _positive(var_hat, "var_hat")
    return float(np.mean((vt - vh) ** 2))


def mae_vol(vol_true, vol_hat) -> float:
    """Mean absolute error on volatility (secondary)."""
    vt = _positive(vol_true, "vol_true")
    vh = _positive(vol_hat, "vol_hat")
    return float(np.mean(np.abs(vt - vh)))


def diebold_mariano(loss_diff) -> dict:
    """Diebold-Mariano test on loss differentials (paired-t form).

    ``loss_diff``: per-fold mean QLIKE differentials
    d_k = mean(QLIKE(garch) - QLIKE(champion)) on fold k.
    Positive values favor the champion. Returns the DM statistic
    (= paired t), two-sided p-value, and n. See module docstring for the
    equivalence argument.
    """
    d = np.asarray(loss_diff, dtype=float).ravel()
    if d.size < 2:
        raise ValueError(f"need at least 2 folds, got {d.size}")
    if not np.all(np.isfinite(d)):
        raise ValueError("loss differentials contain NaN or inf")
    n = d.size
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    if se == 0:
        raise ValueError("zero-variance loss differentials; DM undefined")
    stat = mean / se
    p_value = float(2.0 * stats.t.sf(abs(stat), df=n - 1))
    return {"statistic": stat, "p_value": p_value, "n": n, "mean": mean, "se": se}


def t_confidence_interval(d, alpha: float = 0.05) -> dict:
    """Two-sided Student-t CI for the mean of ``d`` (the GO/NO-GO interval)."""
    d = np.asarray(d, dtype=float).ravel()
    if d.size < 2:
        raise ValueError(f"need at least 2 observations, got {d.size}")
    if not np.all(np.isfinite(d)):
        raise ValueError("observations contain NaN or inf")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = d.size
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    return {
        "mean": mean,
        "se": se,
        "df": n - 1,
        "tcrit": tcrit,
        "lower": mean - tcrit * se,
        "upper": mean + tcrit * se,
        "n": n,
    }
