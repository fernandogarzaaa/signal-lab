"""TAILQ metrics: pinball loss, Kupiec and Christoffersen coverage tests, DM.

Pinball (quantile) loss at tau = 0.05 is the strictly consistent
scoring rule for quantile forecasts (the quantile analog of Patton
2011's QLIKE argument for variances); lower is better. It is linear in
the tail, so no single row can dominate a fold.

Coverage tests follow the frozen pre-registration
(docs/TAILQ_PREREGISTRATION.md):
- Kupiec (1995) unconditional coverage, LR_uc ~ chi2(1).
- Christoffersen (1998) independence, LR_ind ~ chi2(1).
- Conditional coverage LR_cc = LR_uc + LR_ind ~ chi2(2).
Pass iff p > 0.05 in each.

Diebold-Mariano: the campaign runs DM on fold-level mean pinball loss
differentials d_1..d_K (K = 5 folds). With one observation per fold the
folds are non-overlapping blocks, so the HAC long-run variance
estimator collapses to the ordinary sample variance of the fold means
and the DM statistic is algebraically identical to the paired
t-statistic dbar / (s_d / sqrt(K)). The paired-t form is reported
directly, as in VOL/HARVIX.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

TAU = 0.05


def pinball(y_true, q_hat, tau: float = TAU) -> np.ndarray:
    """Elementwise pinball (quantile) loss. Lower is better."""
    y = np.asarray(y_true, dtype=float)
    q = np.asarray(q_hat, dtype=float)
    if y.shape != q.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {q.shape}")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(q))):
        raise ValueError("pinball inputs contain NaN or inf; refusing to score")
    diff = y - q
    return diff * (tau - (diff < 0).astype(float))


def mean_pinball(y_true, q_hat, tau: float = TAU) -> float:
    """Mean pinball loss over a set of rows (e.g. one walk-forward fold)."""
    return float(np.mean(pinball(y_true, q_hat, tau=tau)))


def paired_t(d: np.ndarray) -> dict:
    """Mean, SE, 95% two-sided Student-t CI, t-stat and p-value of d."""
    d = np.asarray(d, dtype=float)
    n = d.size
    if n < 2:
        raise ValueError(f"paired_t needs >= 2 observations, got {n}")
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    df = n - 1
    tcrit = float(stats.t.ppf(0.975, df))
    stat = mean / se if se > 0 else 0.0
    p = float(2.0 * (1.0 - stats.t.cdf(abs(stat), df)))
    return {
        "mean": mean, "se": se, "df": df, "n": n,
        "ci_low": mean - tcrit * se, "ci_high": mean + tcrit * se,
        "stat": float(stat), "p": p,
    }


def kupiec_pof(hits: np.ndarray, p0: float = TAU) -> dict:
    """Kupiec (1995) unconditional coverage test. hits in {0, 1}."""
    hits = np.asarray(hits, dtype=int).ravel()
    n = hits.size
    if n == 0:
        raise ValueError("kupiec: no observations")
    if not np.all((hits == 0) | (hits == 1)):
        raise ValueError("kupiec: hits must be 0/1")
    x = int(hits.sum())
    p_hat = x / n
    if p_hat in (0.0, 1.0):
        # Degenerate: all or no violations; LR is defined by the limit.
        stat = -2.0 * (x * np.log(p0) + (n - x) * np.log(1.0 - p0)
                       - (x * np.log(max(p_hat, 1e-300))
                          + (n - x) * np.log(max(1.0 - p_hat, 1e-300))))
    else:
        stat = -2.0 * (x * np.log(p0 / p_hat)
                       + (n - x) * np.log((1.0 - p0) / (1.0 - p_hat)))
    stat = float(max(stat, 0.0))
    p = float(stats.chi2.sf(stat, 1))
    return {"stat": stat, "p": p, "n": n, "x": x, "rate": p_hat,
            "pass": bool(p > 0.05)}


def christoffersen_independence(hits: np.ndarray) -> dict:
    """Christoffersen (1998) independence test on the hit sequence."""
    hits = np.asarray(hits, dtype=int).ravel()
    n = hits.size
    if n < 2:
        raise ValueError("christoffersen: need >= 2 observations")
    if not np.all((hits == 0) | (hits == 1)):
        raise ValueError("christoffersen: hits must be 0/1")
    prev = hits[:-1]
    cur = hits[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    n0 = n00 + n01
    n1 = n10 + n11
    pi = (n01 + n11) / n
    pi0 = n01 / n0 if n0 > 0 else 0.0
    pi1 = n11 / n1 if n1 > 0 else 0.0

    def _loglik(counts, probs):
        ll = 0.0
        for c, pr in zip(counts, probs):
            if c > 0:
                if pr <= 0:
                    return -np.inf
                ll += c * np.log(pr)
        return ll

    ll_null = _loglik((n00 + n10, n01 + n11), (1.0 - pi, pi))
    ll_alt = _loglik((n00, n01, n10, n11),
                     (1.0 - pi0, pi0, 1.0 - pi1, pi1))
    stat = float(max(-2.0 * (ll_null - ll_alt), 0.0))
    p = float(stats.chi2.sf(stat, 1))
    return {"stat": stat, "p": p, "n": n,
            "counts": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
            "pass": bool(p > 0.05)}


def conditional_coverage(hits: np.ndarray, p0: float = TAU) -> dict:
    """Christoffersen conditional coverage: LR_cc = LR_uc + LR_ind ~ chi2(2)."""
    uc = kupiec_pof(hits, p0=p0)
    ind = christoffersen_independence(hits)
    stat = uc["stat"] + ind["stat"]
    p = float(stats.chi2.sf(stat, 2))
    return {"stat": float(stat), "p": p, "pass": bool(p > 0.05),
            "lr_uc": uc["stat"], "lr_ind": ind["stat"]}
