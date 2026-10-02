"""EVENTVOL campaign: event-window realized volatility vs implied volatility.

Pre-registered in docs/EVENTVOL_PREREGISTRATION.md (frozen, binding).

Target: realized_vol_5d (sample std, ddof=1, of daily log returns over
(t0, t0+5], decimal, daily) on earnings-event rows, one row per
(ticker, earnings t0). Primary baseline: iv_proxy, the market's
forward-looking forecast proxied by beta(i,t0) x VIX(t0)/100/sqrt(252)
(per-stock historical IV is not obtainable from yfinance; the
reformulation is documented as a deliberate design choice in the
pre-registration). Secondary baselines: garch11 (scale-corrected 5-day
term structure, reused from signal_lab.vol), har_vix (ridge on HAR +
VIX features). Challenger: LightGBM on HAR + VIX + event features
(Ederington-Lee IV creation/resolution cycle proxies). Primary metric:
QLIKE on variance. Validation: 5-fold expanding purged walk-forward,
seed 7, 5-day embargo. GO needs the challenger to beat iv_proxy on
paired QLIKE (95% t CI lower bound > 0) AND survive the 95% Model
Confidence Set.

Submodules:
    earnings  yfinance earnings calendar download/cache + event frame
    vix       ^VIX / ^VIX3M download/cache + point-in-time lookup
    features  HAR+VIX and challenger feature sets (point-in-time)
    iv_proxy  beta-scaled VIX baseline forecasts
    mcs       Hansen-Lunde-Nason Model Confidence Set (T_R, day-block
              stationary bootstrap)
    run_campaign  CLI: build-frame | evaluate | confirm-test-eval
"""

__all__ = [
    "earnings",
    "vix",
    "features",
    "iv_proxy",
    "mcs",
    "run_campaign",
]
