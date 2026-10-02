"""Volatility prediction campaign (VOL): pre-registered research program.

Target: realized_vol_5d (sample std, ddof=1, of daily log returns over
(t0, t0+5]). Baseline: GARCH(1,1) by MLE on trailing 252 trading days.
Primary arms: ridge regression and LightGBM on HAR-style point-in-time
price features. Primary metric: QLIKE on variance. Decision rule and
frozen periods live in docs/VOL_PREREGISTRATION.md, which is binding.

Submodules:
    universe  S&P 100 + SPY frozen ticker list
    prices    yfinance OHLCV download, cache, fail-loud quality gates
    frame     (ticker, trading day) frame with t0/t1/published_at/target
    features  HAR-style point-in-time features (+ SPY legs)
    garch     GARCH(1,1) MLE baseline with analytic 5-day term structure
    models    ridge and LightGBM price arms (hyperparameters pre-declared)
    metrics   QLIKE, MSE-on-variance, MAE-on-vol, Diebold-Mariano
    run_campaign  CLI: build-frame | select | evaluate | confirm-test-eval | text-attempt
"""

__all__ = [
    "universe",
    "prices",
    "frame",
    "features",
    "garch",
    "models",
    "metrics",
    "run_campaign",
]
