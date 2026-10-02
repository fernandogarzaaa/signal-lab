"""Model Confidence Set (Hansen, Lunde, Nason 2011) for EVENTVOL.

Procedure: on the per-row QLIKE loss matrix L (n rows x m models),
iteratively test the null of equal predictive ability over the current
model set M with the range statistic

    T_R,M = max_{i,j in M} |tbar_ij| / sqrt(var(tbar_ij)),

where tbar_ij is the mean loss differential between models i and j.
The variance and the null distribution come from a day-block
stationary bootstrap (mean block length 10 trading days, B = 5000):
whole trading days are resampled so the cross-section is preserved.
If the bootstrap p-value < alpha (0.05), the worst model
e = argmax_i tbar_i. / sqrt(var(tbar_i.)) is eliminated and the test
repeats on the survivors. The final set is the (1 - alpha) MCS.

Documented limitation (docs/EVENTVOL_PREREGISTRATION.md): with 5
walk-forward folds the pooled rows carry within-fold dependence the
day-block bootstrap only partly captures; the MCS is a GO conjunct,
not the sole gate.

Fail loud: NaN/inf losses, fewer than 2 rows, or fewer than 2 models
raise. Deterministic given ``seed``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MCS_ALPHA = 0.05
MCS_BOOT = 5000
MCS_MEAN_BLOCK = 10
MCS_SEED = 7


def _day_groups(t0: pd.Series) -> list[np.ndarray]:
    days = pd.to_datetime(t0).dt.normalize()
    groups: list[np.ndarray] = []
    for _, idx in days.groupby(days).groups.items():
        groups.append(np.asarray(sorted(idx), dtype=np.int64))
    return groups


def _stationary_day_indices(day_groups: list[np.ndarray], n: int,
                            mean_block: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Stationary-bootstrap row indices: resample whole trading days."""
    n_days = len(day_groups)
    parts: list[np.ndarray] = []
    total = 0
    while total < n:
        start = int(rng.integers(n_days))
        length = int(rng.geometric(1.0 / mean_block))
        for k in range(length):
            g = day_groups[(start + k) % n_days]
            parts.append(g)
            total += len(g)
            if total >= n:
                break
    return np.concatenate(parts)[:n]


def mcs(loss: pd.DataFrame,
        t0: pd.Series,
        alpha: float = MCS_ALPHA,
        n_boot: int = MCS_BOOT,
        mean_block: int = MCS_MEAN_BLOCK,
        seed: int = MCS_SEED,
        log=print) -> dict:
    """Run the Model Confidence Set elimination.

    ``loss``: DataFrame (n rows x m models) of per-row losses, finite.
    ``t0``: per-row dates (length n) defining the day blocks.
    Returns dict with keys: survivors (list[str]), eliminated
    (list of (model, p_value) in elimination order), mcs_p (dict
    model -> p-value of the round in which it was eliminated, or the
    final round's p for survivors), n_rows, n_boot, alpha.
    """
    if not isinstance(loss, pd.DataFrame):
        raise ValueError("[mcs] loss must be a DataFrame")
    models = list(loss.columns)
    m = len(models)
    n = len(loss)
    if m < 2:
        raise ValueError(f"[mcs] need >= 2 models, got {m}")
    if n < 2:
        raise ValueError(f"[mcs] need >= 2 rows, got {n}")
    if len(t0) != n:
        raise ValueError("[mcs] t0 length must match loss rows")
    L = loss.to_numpy(dtype=float)
    if not np.all(np.isfinite(L)):
        raise ValueError("[mcs] losses contain NaN or inf; refusing")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"[mcs] alpha must be in (0, 1), got {alpha}")

    day_groups = _day_groups(t0)
    rng = np.random.default_rng(seed)
    surviving = list(models)
    eliminated: list[tuple[str, float]] = []
    mcs_p: dict[str, float] = {}

    while True:
        cols = [models.index(x) for x in surviving]
        Lm = L[:, cols]
        mm = len(surviving)
        # D[b] is (mm, mm): bootstrap mean loss differentials.
        D = Lm[:, :, None] - Lm[:, None, :]  # (n, mm, mm)
        dbar = D.mean(axis=0)
        boot = np.empty((n_boot, mm, mm))
        for b in range(n_boot):
            idx = _stationary_day_indices(day_groups, n, mean_block, rng)
            boot[b] = D[idx].mean(axis=0)
        var = boot.var(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = np.where(var > 0, dbar / np.sqrt(var), 0.0)
        stat = float(np.max(np.abs(tstat)))
        with np.errstate(divide="ignore", invalid="ignore"):
            tboot = np.where(
                var > 0, np.abs(boot - dbar) / np.sqrt(var), 0.0)
        stat_boot = tboot.reshape(n_boot, -1).max(axis=1)
        p_value = float((1.0 + np.sum(stat_boot >= stat)) / (n_boot + 1.0))
        log(f"[mcs] models={surviving} T_R={stat:.4f} p={p_value:.4f}")
        for x in surviving:
            mcs_p[x] = p_value
        if p_value >= alpha or mm == 1:
            break
        # Eliminate the worst model: max standardized average differential.
        dbar_i = dbar.mean(axis=1)
        boot_i = boot.mean(axis=2)
        var_i = boot_i.var(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_i = np.where(var_i > 0, dbar_i / np.sqrt(var_i),
                           np.where(dbar_i > 0, np.inf, 0.0))
        e = int(np.argmax(t_i))
        eliminated.append((surviving[e], p_value))
        log(f"[mcs] eliminated {surviving[e]} (p={p_value:.4f})")
        del surviving[e]

    return {
        "survivors": surviving,
        "eliminated": eliminated,
        "mcs_p": mcs_p,
        "n_rows": n,
        "n_models": m,
        "n_boot": n_boot,
        "alpha": alpha,
        "seed": seed,
    }
