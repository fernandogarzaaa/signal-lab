"""Deflated Sharpe Ratio (DSR) as a model-selection honesty metric.

Reference: Bailey, D. H. & Lopez de Prado, M. (2014), "The Deflated
Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and
Non-Normality", Journal of Portfolio Management.

When K model variants are tried and the best one is reported, its
Sharpe ratio is optimistic: it is the maximum of K draws, not a single
draw. The DSR answers "what is the probability the true Sharpe ratio is
positive, after accounting for how many variants were tried and for
non-normal returns?"

    DSR = Phi( (SR_hat - SR_0) * sqrt(T - 1)
               / sqrt(1 - skew * SR_hat + (kurt - 1) / 4 * SR_hat**2) )

    SR_0 = sqrt(V) * ((1 - gamma) * Phi^{-1}(1 - 1 / N)
                      + gamma * Phi^{-1}(1 - 1 / (N * e)))

where SR_hat is the observed (annualized) Sharpe of the selected
variant, T the number of return observations, skew/kurt the sample
skewness and Pearson kurtosis of its returns, N the number of variants
tried, V the sample variance of the N trial Sharpe ratios, gamma the
Euler-Mascheroni constant, and Phi the standard normal CDF.

Usage contract (reported, NOT a gate):
- ``dsr_report`` takes {variant_name: returns} for every variant that
  was tried (including losers) and returns per-variant Sharpe, DSR,
  and a ``likely_false_discovery`` flag (DSR < 0.95).
- A low DSR does not fail anything; it is disclosed next to the
  Sharpe so selection bias is visible.

Cross-check note (APP-006): the purgedcv reference
(eslazarev/purged-cross-validation, v0.1.10) implements PBO but not
DSR; this module implements DSR directly from the paper above. Our
validation harness (``signal_lab.validation``) stays canonical.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def _as_returns(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float).ravel()
    if r.size < 2:
        raise ValueError(f"need at least 2 return observations, got {r.size}")
    if not np.all(np.isfinite(r)):
        raise ValueError("returns contain NaN or inf; refusing to score")
    return r


def sharpe_ratio(returns, freq: int = 252) -> float:
    """Annualized Sharpe ratio (zero risk-free rate)."""
    if freq < 1:
        raise ValueError(f"freq must be >= 1, got {freq}")
    r = _as_returns(returns)
    sd = r.std(ddof=1)
    if sd == 0:
        raise ValueError("returns have zero variance; Sharpe is undefined")
    return float(r.mean() / sd * math.sqrt(freq))


def expected_sharpe_null(trial_srs) -> float:
    """SR_0: expected Sharpe of the best of N trials under the null.

    ``trial_srs``: Sharpe ratios of ALL tried variants (winners and
    losers). V is their sample variance (ddof=1, the unbiased estimator).
    """
    srs = np.asarray(trial_srs, dtype=float).ravel()
    if srs.size < 2:
        raise ValueError(
            f"need Sharpe ratios of at least 2 tried variants, got {srs.size}"
        )
    if not np.all(np.isfinite(srs)):
        raise ValueError("trial Sharpe ratios contain NaN or inf")
    n = srs.size
    var = srs.var(ddof=1)
    if var < 0:
        var = 0.0  # numerical guard; var() is non-negative by construction
    term = ((1.0 - EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / n)
            + EULER_MASCHERONI * stats.norm.ppf(1.0 - 1.0 / (n * math.e)))
    return float(math.sqrt(var) * term)


def deflated_sharpe_ratio(
    returns,
    trial_srs,
    freq: int = 252,
) -> float:
    """Deflated Sharpe Ratio of one variant's returns given all trials.

    ``returns``: per-period returns of the variant being scored.
    ``trial_srs``: Sharpe ratios (same ``freq``) of every variant tried.
    Returns P(true Sharpe > 0), corrected for selection bias over
    ``len(trial_srs)`` trials and for skew/kurtosis of ``returns``.
    """
    r = _as_returns(returns)
    sr_hat = sharpe_ratio(r, freq=freq)
    sr_0 = expected_sharpe_null(trial_srs)
    t = r.size
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))  # Pearson kurtosis
    denom = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2
    if denom <= 0:
        raise ValueError(
            f"non-positive DSR denominator ({denom}); returns too pathological to score"
        )
    stat = (sr_hat - sr_0) * math.sqrt(t - 1) / math.sqrt(denom)
    return float(stats.norm.cdf(stat))


def dsr_report(
    variant_returns: dict[str, object],
    freq: int = 252,
    discovery_threshold: float = 0.95,
) -> pd.DataFrame:
    """Report Sharpe, DSR, and false-discovery flags for tried variants.

    Every variant tried must appear (winners AND losers): hiding losers
    understates N and inflates the DSR, which is exactly the bias this
    metric exists to expose. The report is informational; nothing fails.
    """
    if len(variant_returns) < 2:
        raise ValueError("dsr_report needs at least 2 tried variants")
    names = list(variant_returns.keys())
    srs = [sharpe_ratio(variant_returns[n], freq=freq) for n in names]
    rows = []
    for name, sr in zip(names, srs):
        dsr = deflated_sharpe_ratio(variant_returns[name], srs, freq=freq)
        rows.append({
            "variant": name,
            "sharpe": sr,
            "dsr": dsr,
            "likely_false_discovery": bool(dsr < discovery_threshold),
        })
    out = pd.DataFrame(rows).sort_values("dsr", ascending=False).reset_index(drop=True)
    return out
