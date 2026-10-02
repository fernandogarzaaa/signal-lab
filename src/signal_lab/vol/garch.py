"""GARCH(1,1) baseline: MLE on trailing 252 trading days, analytic term structure.

Frozen spec (docs/VOL_PREREGISTRATION.md):
- Fit by maximum likelihood (Gaussian innovations, zero-mean: the daily
  mean is negligible next to daily variance and the spec's recursion is
  written in r_t^2) on the trailing 252 trading days of daily log returns
  through t0.
- Forecast = analytic 5-day term structure
      E[sigma^2_{t+h}|F_t] = sbar^2 + (alpha+beta)^(h-1) * (s2_1 - sbar^2),
  averaged over h = 1..5 and square-rooted to the target's DAILY scale,
  where sbar^2 = omega / (1 - alpha - beta) and
  s2_1 = omega + alpha * r_t^2 + beta * s2_t (one-step-ahead from the last
  fitted conditional variance).
  DEVIATION (logged in docs/VOL_RESULTS.md): the pre-registration says
  "summed over h = 1..5, square-rooted", which literally is the 5-day
  cumulative vol, sqrt(5) ~= 2.24x larger than the frozen target
  (realized_vol_5d = sample std of daily log returns, "decimal, daily").
  The mean (not sum) puts the baseline on the target's scale, matching
  the naive arm; the literal reading would cripple the baseline by
  construction and void the GO/NO-GO comparison.
- Fit failures (non-convergence, alpha + beta >= 1, non-finite or
  non-positive variances, short estimation window) fall back to the naive
  persistence forecast and are COUNTED, never hidden.

Implemented with scipy.optimize L-BFGS-B and an analytic gradient (no new
dependencies; ``arch`` is not installed and CI installs only from
packages/cli/engine/requirements.txt).
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ESTIMATION_WINDOW = 252
HORIZON_DAYS = 5

# Two pre-declared starting points (variance-targeting style). A second
# start guards against local-optimum non-convergence; both are fixed, not
# tuned.
_STARTS = ((0.05, 0.07, 0.88), (0.10, 0.12, 0.80))  # (omega/var, alpha, beta)


@dataclass(frozen=True)
class GarchFit:
    omega: float
    alpha: float
    beta: float
    sigma2_last: float  # conditional variance of the last in-sample return
    n: int
    converged: bool
    detail: str  # "ok" or the failure reason; failures are counted by callers


def _recursion(theta: np.ndarray, r2: np.ndarray, var0: float):
    """Conditional variances and their derivatives wrt (omega, alpha, beta).

    sigma2[0] = var0 (fixed); for t >= 1:
    sigma2[t] = omega + alpha * r2[t-1] + beta * sigma2[t-1].
    """
    omega, alpha, beta = theta
    n = r2.shape[0]
    s2 = np.empty(n)
    d_om = np.empty(n)
    d_al = np.empty(n)
    d_be = np.empty(n)
    s2[0] = var0
    d_om[0] = d_al[0] = d_be[0] = 0.0
    for t in range(1, n):
        s2[t] = omega + alpha * r2[t - 1] + beta * s2[t - 1]
        d_om[t] = 1.0 + beta * d_om[t - 1]
        d_al[t] = r2[t - 1] + beta * d_al[t - 1]
        d_be[t] = s2[t - 1] + beta * d_be[t - 1]
    return s2, d_om, d_al, d_be


def _negloglik_and_grad(theta: np.ndarray, r: np.ndarray, r2: np.ndarray,
                        var0: float):
    s2, d_om, d_al, d_be = _recursion(theta, r2, var0)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return np.inf, np.zeros(3)
    inv = 1.0 / s2
    # d loglik / d sigma2[t] = -0.5 * (1/s2 - r^2/s2^2); negate for the NLL.
    w = 0.5 * (inv - r * r * inv * inv)
    nll = 0.5 * float(np.sum(np.log(2.0 * np.pi) + np.log(s2) + r * r * inv))
    grad = np.array([
        float(np.sum(w * d_om)),
        float(np.sum(w * d_al)),
        float(np.sum(w * d_be)),
    ])
    if not np.all(np.isfinite(grad)):
        return np.inf, np.zeros(3)
    return nll, grad


def _fail(detail: str, n: int) -> GarchFit:
    return GarchFit(omega=np.nan, alpha=np.nan, beta=np.nan,
                    sigma2_last=np.nan, n=n, converged=False, detail=detail)


def fit_garch11(r: np.ndarray, window: int = ESTIMATION_WINDOW) -> GarchFit:
    """MLE fit of GARCH(1,1) on the last ``window`` returns of ``r``."""
    r = np.asarray(r, dtype=float).ravel()
    if r.size < window:
        return _fail(f"short-window:{r.size}<{window}", r.size)
    rw = r[-window:]
    r2 = rw * rw
    if not np.all(np.isfinite(rw)):
        return _fail("non-finite-returns", window)
    var0 = float(np.var(rw, ddof=1))
    if not np.isfinite(var0) or var0 <= 0:
        return _fail("degenerate-variance", window)

    bounds = [(1e-12, 10.0 * var0), (1e-8, 0.999), (1e-8, 0.999)]
    best = None
    for om_frac, a0, b0 in _STARTS:
        x0 = np.array([om_frac * var0, a0, b0])
        try:
            res = minimize(
                _negloglik_and_grad, x0, jac=True, args=(rw, r2, var0),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 500},
            )
        except Exception:  # optimizer-level blowup -> try next start
            continue
        if not res.success or not np.isfinite(res.fun):
            continue
        omega, alpha, beta = (float(v) for v in res.x)
        if alpha + beta >= 1.0:
            continue
        if best is None or res.fun < best[0]:
            best = (float(res.fun), omega, alpha, beta)
    if best is None:
        return _fail("non-convergence", window)
    _, omega, alpha, beta = best
    s2, _, _, _ = _recursion(np.array([omega, alpha, beta]), r2, var0)
    sigma2_last = float(s2[-1])
    if not np.isfinite(sigma2_last) or sigma2_last <= 0:
        return _fail("non-positive-variance", window)
    return GarchFit(omega=omega, alpha=alpha, beta=beta,
                    sigma2_last=sigma2_last, n=window,
                    converged=True, detail="ok")


def forecast_garch11(fit: GarchFit, r_last: float,
                     horizon: int = HORIZON_DAYS) -> float:
    """Analytic h-step term-structure forecast, square-rooted to vol.

    Returns sqrt(mean_h E[sigma^2_{t+h}|F_t]): the forecast of expected
    DAILY variance over the window, square-rooted to the target's daily
    scale. DEVIATION from the pre-registration's literal "summed over
    h=1..5, square-rooted" (which is sqrt of the SUM = 5-day cumulative
    vol, sqrt(5) ~= 2.24x the defined target scale): the frozen target
    realized_vol_5d is "sample std (ddof=1) of daily log returns ...,
    decimal, daily" (docs/TARGETS_AND_FEATURES.md v2.0), and the naive
    baseline (trailing realized_vol_5d) is on that daily scale too. The
    literal reading would cripple the baseline by construction and make
    the GO/NO-GO comparison meaningless; the deviation is logged with
    rationale in docs/VOL_RESULTS.md.
    """
    if not fit.converged:
        raise ValueError(f"cannot forecast from a failed fit ({fit.detail})")
    if horizon != HORIZON_DAYS:
        raise ValueError(f"frozen horizon is {HORIZON_DAYS}, got {horizon}")
    persistence = fit.alpha + fit.beta
    sbar2 = fit.omega / (1.0 - persistence)
    s2_1 = fit.omega + fit.alpha * r_last ** 2 + fit.beta * fit.sigma2_last
    total = 0.0
    for h in range(1, horizon + 1):
        total += sbar2 + persistence ** (h - 1) * (s2_1 - sbar2)
    mean_var = total / horizon
    if not np.isfinite(mean_var) or mean_var <= 0:
        raise ValueError(f"non-positive forecast variance {mean_var}")
    return math.sqrt(mean_var)


def naive_forecast(returns: np.ndarray, t0_pos: int,
                   horizon: int = HORIZON_DAYS) -> float:
    """Persistence baseline: trailing realized_vol_5d as of t0.

    Uses the 5 fully-observed daily returns ending at t0: r_{t0-4..t0}.
    """
    window = returns[t0_pos - horizon + 1:t0_pos + 1]
    if window.shape[0] < horizon or not np.all(np.isfinite(window)):
        return np.nan
    return float(np.std(window, ddof=1))


def forecast_at_t0(returns: np.ndarray, t0_pos: int,
                   window: int = ESTIMATION_WINDOW,
                   horizon: int = HORIZON_DAYS) -> tuple[float, bool]:
    """GARCH(1,1) forecast for one t0; (forecast, fell_back_to_naive).

    ``returns``: full daily log-return array for the ticker; ``t0_pos``:
    index of t0 in that array (estimation uses returns through t0).
    """
    r_last = float(returns[t0_pos])
    naive = naive_forecast(returns, t0_pos, horizon)
    fit = fit_garch11(returns[:t0_pos + 1], window=window)
    if not fit.converged:
        return naive, True
    try:
        fc = forecast_garch11(fit, r_last, horizon)
    except ValueError:
        return naive, True
    if not np.isfinite(fc) or fc <= 0:
        return naive, True
    return fc, False


def _ticker_task(args) -> list[tuple[float, bool]]:
    """One task per ticker: forecasts for all its t0s. Pickle-friendly."""
    dates_ord, returns, t0_ords, window, horizon = args
    out: list[tuple[float, bool]] = []
    for t0o in t0_ords:
        pos = int(np.searchsorted(dates_ord, t0o, side="left"))
        if pos >= len(dates_ord) or dates_ord[pos] != t0o:
            raise ValueError(f"t0 ordinal {t0o} not on the ticker's date grid")
        out.append(forecast_at_t0(returns, pos, window=window, horizon=horizon))
    return out


def _panel_cache_key(rows: pd.DataFrame, window: int, horizon: int) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(f"{window}/{horizon}/".encode())
    for t, d in zip(rows["ticker"].astype(str), rows["t0"].astype(str)):
        h.update(f"{t}@{d};".encode())
    return h.hexdigest()[:32]


def forecast_panel(rows: pd.DataFrame,
                   returns_by_ticker: dict[str, pd.Series],
                   window: int = ESTIMATION_WINDOW,
                   horizon: int = HORIZON_DAYS,
                   n_jobs: int = 1,
                   cache_path=None,
                   log=print) -> pd.DataFrame:
    """GARCH(1,1) forecasts for every (ticker, t0) in ``rows``.

    Returns a DataFrame aligned to ``rows`` with columns
    ``garch_vol`` and ``garch_fallback``. Deterministic regardless of
    ``n_jobs`` (per-ticker tasks reassembled in input order).

    ``cache_path`` (optional): parquet file caching the panel. The cache
    is keyed by the exact (ticker, t0) row order plus window/horizon; a
    matching cache is reused, otherwise it is recomputed and rewritten.
    The cache is a local resume aid only (never committed).

    When ``cache_path`` is given, per-ticker progress is checkpointed to
    ``<cache>.checkpoint.parquet`` after every ticker, so an interrupted
    run (restart, kill) resumes where it left off instead of redoing
    every fit. The checkpoint is deleted once the full panel is cached.
    """
    from concurrent.futures import as_completed

    rows = rows.reset_index(drop=True)
    ckpt_path = None
    if cache_path is not None:
        from pathlib import Path
        cache_path = Path(cache_path)
        key_path = cache_path.with_suffix(".key")
        ckpt_path = cache_path.with_name(cache_path.stem + ".checkpoint.parquet")
        key = _panel_cache_key(rows, window, horizon)
        if cache_path.exists() and key_path.exists():
            if key_path.read_text().strip() == key:
                cached = pd.read_parquet(cache_path)
                if (len(cached) == len(rows)
                        and (cached["ticker"].astype(str).to_numpy()
                             == rows["ticker"].astype(str).to_numpy()).all()
                        and (cached["t0"].astype(str).to_numpy()
                             == rows["t0"].astype(str).to_numpy()).all()):
                    n_fb = int(cached["garch_fallback"].sum())
                    log(f"[garch] reused cached panel ({len(cached)} rows, "
                        f"{n_fb} fallbacks)")
                    return cached
                log("[garch] cache key matched but rows differ; recomputing")
            else:
                log("[garch] cache key mismatch; recomputing panel")
    t0_dates = pd.to_datetime(rows["t0"]).dt.date
    by_ticker: dict[str, list[int]] = {}
    for i, (t, d) in enumerate(zip(rows["ticker"].astype(str), t0_dates)):
        by_ticker.setdefault(t, []).append(i)
    expected_t0s = {t: [str(t0_dates[i]) for i in idxs]
                    for t, idxs in by_ticker.items()}

    work: list[tuple[str, list[int], tuple]] = []
    for t, idxs in by_ticker.items():
        if t not in returns_by_ticker:
            raise ValueError(f"no return series for ticker {t}")
        rs = returns_by_ticker[t].sort_index()
        dates_ord = np.array([d.toordinal() for d in rs.index.date], dtype=np.int64)
        rets = rs.to_numpy(dtype=float)
        t0_ords = np.array([d.toordinal() for d in t0_dates.iloc[idxs]], dtype=np.int64)
        work.append((t, idxs, (dates_ord, rets, t0_ords, window, horizon)))

    # --- resume from per-ticker checkpoint ---
    completed: dict[str, list[tuple[float, bool]]] = {}
    if ckpt_path is not None and ckpt_path.exists():
        try:
            ck = pd.read_parquet(ckpt_path)
            for t, g in ck.groupby("ticker"):
                t = str(t)
                if t in by_ticker and g["t0"].astype(str).tolist() == expected_t0s[t]:
                    completed[t] = list(zip(g["garch_vol"].astype(float).tolist(),
                                            g["garch_fallback"].astype(bool).tolist()))
            log(f"[garch] checkpoint: {len(completed)}/{len(work)} tickers already done")
        except Exception as exc:
            log(f"[garch] checkpoint unreadable ({exc}); recomputing")

    def _write_checkpoint() -> None:
        if ckpt_path is None or not completed:
            return
        parts = [
            pd.DataFrame({
                "ticker": t,
                "t0": expected_t0s[t],
                "garch_vol": [fc for fc, _ in completed[t]],
                "garch_fallback": [fb for _, fb in completed[t]],
            })
            for t in completed
        ]
        pd.concat(parts, ignore_index=True).to_parquet(ckpt_path, index=False)

    pending = [w for w in work if w[0] not in completed]
    log(f"[garch] {len(pending)} tickers to forecast ({len(completed)} resumed)")
    if n_jobs == 1:
        for w in pending:
            t, _, args = w
            completed[t] = _ticker_task(args)
            _write_checkpoint()
            n_done = len(completed)
            if n_done % 10 == 0 or n_done == len(work):
                log(f"[garch] {n_done}/{len(work)} tickers")
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            future_to_ticker = {ex.submit(_ticker_task, w[2]): w[0] for w in pending}
            for fut in as_completed(future_to_ticker):
                t = future_to_ticker[fut]
                completed[t] = fut.result()  # raises loudly on worker error
                _write_checkpoint()
                n_done = len(completed)
                if n_done % 10 == 0 or n_done == len(work):
                    log(f"[garch] {n_done}/{len(work)} tickers")

    vols = np.empty(len(rows))
    fallbacks = np.empty(len(rows), dtype=bool)
    for t, idxs, _ in work:
        res = completed[t]
        for i, (fc, fb) in zip(idxs, res):
            vols[i] = fc
            fallbacks[i] = fb
    n_fb = int(fallbacks.sum())
    n_nan = int(np.isnan(vols).sum())
    if n_nan:
        raise ValueError(
            f"[garch] {n_nan} NaN forecasts (no naive value available); "
            "refusing to score NaN forecasts"
        )
    log(f"[garch] {len(rows)} forecasts, {n_fb} naive fallbacks "
        f"({100.0 * n_fb / max(len(rows), 1):.2f}%)")
    out = rows[["ticker", "t0"]].copy()
    out["garch_vol"] = vols
    out["garch_fallback"] = fallbacks
    if cache_path is not None:
        out.to_parquet(cache_path, index=False)
        key_path.write_text(_panel_cache_key(rows, window, horizon))
        if ckpt_path is not None and ckpt_path.exists():
            ckpt_path.unlink()
        log(f"[garch] panel cached -> {cache_path}")
    return out
