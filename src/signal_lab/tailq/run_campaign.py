"""TAILQ campaign runner: dev walk-forward evaluation (and GO-gated test eval).

Pipeline:
1. Load universe (frozen data/universe_vol.csv), prices (yfinance cache),
   earnings (yfinance earnings calendar cache), VIX (^VIX/^VIX3M cache).
2. Build the TAILQ event frame (3-day event return target), keep dev
   rows (t0 <= 2026-06-30), build the 15 frozen features.
3. 5-fold expanding purged walk-forward on t0
   (WalkForwardConfig: horizon_days=3, embargo_days=3, seed=7,
   min_train=50). check_no_confirmation enforced on every build.
4. Per fold: fit the LightGBM quantile challenger on train rows;
   forecast test rows for challenger, caviar, garch_hs, naive.
   CAViaR/GARCH fit failures fall back to naive and are COUNTED.
5. Verdict: mean paired pinball differential d = pinball(caviar) -
   pinball(challenger); GO conjuncts from the frozen pre-registration;
   Kupiec + Christoffersen coverage on pooled dev test rows;
   95% MCS over the arms (reported, not a gate); DSR honesty metric.

Caches (data/*.parquet) are local resume aids and are never committed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from signal_lab.eventvol.mcs import mcs
from signal_lab.stats import deflated_sharpe as dsr_mod
from signal_lab.tailq import caviar as caviar_mod
from signal_lab.tailq import features as feat_mod
from signal_lab.tailq import frame as frame_mod
from signal_lab.tailq import garch_hs as ghs_mod
from signal_lab.tailq import metrics as met
from signal_lab.tailq import models as models_mod
from signal_lab.validation import periods
from signal_lab.validation.splits import WalkForwardConfig, make_splits
from signal_lab.vol import universe as universe_mod
from signal_lab.vol.prices import load_or_download_prices

PRICE_CACHE = "data/vol_prices.parquet"
VIX_CACHE = "data/harvix_vix.parquet"
UNIVERSE_CSV = "data/universe_vol.csv"

ARMS = ["lgbm_quantile", "caviar", "garch_hs", "naive"]


def _ticker_return_panels(prices: pd.DataFrame) -> dict:
    """Per ticker: (date ordinals, full log-return array with rets[0]=NaN)."""
    panels = {}
    for t, g in prices.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        ords = pd.to_datetime(g["date"]).dt.date.apply(
            lambda d: d.toordinal()).to_numpy()
        closes = g["adj_close"].to_numpy(dtype=float)
        rets = np.full_like(closes, np.nan)
        rets[1:] = np.log(closes[1:] / closes[:-1])
        panels[str(t)] = (ords, rets)
    return panels


def _forecast_row(ticker: str, t0, panels: dict) -> dict:
    """Forecast all history-based arms for one (ticker, t0)."""
    ords, rets = panels[ticker]
    pos = int(np.searchsorted(ords, pd.Timestamp(t0).date().toordinal(),
                              side="left"))
    if pos >= len(ords) or ords[pos] != pd.Timestamp(t0).date().toordinal():
        raise ValueError(f"[tailq] t0 {t0} not on calendar for {ticker}")
    out: dict = {}
    fb: dict = {}
    q_c, fb_c = caviar_mod.forecast_at_t0(rets, pos)
    out["caviar"], fb["caviar"] = q_c, fb_c
    q_g, fb_g = ghs_mod.forecast_garch_hs_3d(rets, pos)
    out["garch_hs"], fb["garch_hs"] = q_g, fb_g
    rw = rets[:pos + 1]
    out["naive"] = models_mod.naive_3d_quantile(rw)
    return out, fb


def run_dev_eval(prices: pd.DataFrame,
                 events: pd.DataFrame,
                 vix: pd.DataFrame,
                 pooled_cache: str | None = None,
                 log=print) -> dict:
    """Full dev walk-forward evaluation. Returns the verdict dict."""
    dev = frame_mod.dev_frame(events)
    feat = feat_mod.build_features(dev, prices, vix, log=log)
    log(f"[tailq] dev rows for evaluation: {len(feat)}")
    X = feat[feat_mod.FEATURE_COLUMNS].to_numpy(dtype=float)
    y = feat["target"].to_numpy(dtype=float)

    trading_days = np.sort(pd.to_datetime(prices["date"]).dt.date.apply(
        lambda d: d.toordinal()).unique())
    cfg = WalkForwardConfig(n_splits=5, window="expanding",
                            horizon_days=3, embargo_days=3,
                            min_train=50, retrain_every=1,
                            seed=7).validated()
    splits = make_splits(feat, cfg, trading_days)
    panels = _ticker_return_panels(prices)

    fold_rows = []
    pooled: list[dict] = []
    caviar_fallbacks = 0
    garch_fallbacks = 0
    for s in splits:
        tri, tei = s["train_idx"], s["test_idx"]
        booster = models_mod.fit_lgbm_quantile(X[tri], y[tri])
        q_ch = models_mod.predict_lgbm_quantile(booster, X[tei])
        test = feat.iloc[tei].reset_index(drop=True)
        q_cv = np.empty(len(test))
        q_gh = np.empty(len(test))
        q_nv = np.empty(len(test))
        for i, row in test.iterrows():
            f, fb = _forecast_row(str(row["ticker"]), row["t0"], panels)
            q_cv[i], q_gh[i], q_nv[i] = f["caviar"], f["garch_hs"], f["naive"]
            caviar_fallbacks += int(fb["caviar"])
            garch_fallbacks += int(fb["garch_hs"])
        pin = {
            "lgbm_quantile": met.mean_pinball(y[tei], q_ch),
            "caviar": met.mean_pinball(y[tei], q_cv),
            "garch_hs": met.mean_pinball(y[tei], q_gh),
            "naive": met.mean_pinball(y[tei], q_nv),
        }
        fold_rows.append({
            "fold": s["fold"],
            "test_start": str(s["test_start"]),
            "test_end": str(s["test_end"]),
            "n_train": int(len(tri)),
            "n_test": int(len(tei)),
            "n_purged": int(s["n_purged"]),
            "n_embargoed": int(s["n_embargoed"]),
            **{f"pinball_{a}": pin[a] for a in ARMS},
            "d_caviar_minus_challenger":
                float(pin["caviar"] - pin["lgbm_quantile"]),
        })
        for i, row in test.iterrows():
            pooled.append({
                "t0": row["t0"], "ticker": row["ticker"], "y": float(y[tei][i]),
                "q_challenger": float(q_ch[i]), "q_caviar": float(q_cv[i]),
                "q_garch_hs": float(q_gh[i]), "q_naive": float(q_nv[i]),
            })
        log(f"[tailq] fold {s['fold']}: n_test={len(tei)} "
            f"pinball champ={pin['lgbm_quantile']:.6f} "
            f"caviar={pin['caviar']:.6f} "
            f"d={pin['caviar'] - pin['lgbm_quantile']:+.6f}")

    d = np.array([f["d_caviar_minus_challenger"] for f in fold_rows])
    pt = met.paired_t(d)
    pooled_df = pd.DataFrame(pooled).sort_values(["t0", "ticker"]).reset_index(drop=True)
    hits_ch = (pooled_df["y"] < pooled_df["q_challenger"]).to_numpy(dtype=int)
    hits_cv = (pooled_df["y"] < pooled_df["q_caviar"]).to_numpy(dtype=int)
    cov_ch = {
        "kupiec": met.kupiec_pof(hits_ch),
        "christoffersen_ind": met.christoffersen_independence(hits_ch),
        "conditional": met.conditional_coverage(hits_ch),
    }
    cov_cv = {
        "kupiec": met.kupiec_pof(hits_cv),
        "christoffersen_ind": met.christoffersen_independence(hits_cv),
        "conditional": met.conditional_coverage(hits_cv),
    }
    loss = pd.DataFrame({
        "lgbm_quantile": met.pinball(pooled_df["y"], pooled_df["q_challenger"]),
        "caviar": met.pinball(pooled_df["y"], pooled_df["q_caviar"]),
        "garch_hs": met.pinball(pooled_df["y"], pooled_df["q_garch_hs"]),
        "naive": met.pinball(pooled_df["y"], pooled_df["q_naive"]),
    })
    mcs_res = mcs(loss, t0=pooled_df["t0"], seed=7)

    pinball_wins = bool(pt["mean"] > 0 and pt["ci_low"] > 0)
    chall_passes_all = all(v["pass"] for v in cov_ch.values())
    caviar_fails_one = any(not v["pass"] for v in cov_cv.values())
    verdict = ("GO" if (pinball_wins and chall_passes_all and caviar_fails_one)
               else "NO-GO")

    # DSR honesty metric (reported, not a gate): per-row pinball gains
    # over naive as pseudo-returns, freq=1, over all evaluated arms.
    gains = {
        "lgbm_quantile": met.pinball(pooled_df["y"], pooled_df["q_naive"])
                         - met.pinball(pooled_df["y"], pooled_df["q_challenger"]),
        "caviar": met.pinball(pooled_df["y"], pooled_df["q_naive"])
                  - met.pinball(pooled_df["y"], pooled_df["q_caviar"]),
        "garch_hs": met.pinball(pooled_df["y"], pooled_df["q_naive"])
                    - met.pinball(pooled_df["y"], pooled_df["q_garch_hs"]),
    }
    dsr_report = dsr_mod.dsr_report(
        {k: np.asarray(v, dtype=float) for k, v in gains.items()}, freq=1)
    if pooled_cache is not None:
        Path(pooled_cache).parent.mkdir(parents=True, exist_ok=True)
        pooled_df.to_parquet(pooled_cache, index=False)
        log(f"[tailq] pooled forecasts -> {pooled_cache}")

    return {
        "verdict": verdict,
        "n_dev_rows": int(len(feat)),
        "n_pooled_test_rows": int(len(pooled_df)),
        "caviar_fallbacks": int(caviar_fallbacks),
        "garch_hs_fallbacks": int(garch_fallbacks),
        "folds": fold_rows,
        "paired_t": pt,
        "dm_note": ("paired-t form on fold means = DM with HAC collapsing "
                    "on non-overlapping fold blocks; see tailq.metrics"),
        "coverage_challenger": cov_ch,
        "coverage_caviar": cov_cv,
        "mcs": mcs_res,
        "dsr": dsr_report.to_dict("records"),
        "dsr_note": ("per-row pinball gains over naive as pseudo-returns, "
                     "freq=1; reported, not a gate"),
        "go_conjuncts": {
            "pinball_win": pinball_wins,
            "challenger_passes_all_coverage": chall_passes_all,
            "caviar_fails_at_least_one": caviar_fails_one,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="TAILQ dev evaluation")
    ap.add_argument("--out", default="docs/tailq_results.json")
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    tickers = universe_mod.load_universe(UNIVERSE_CSV)
    prices = load_or_download_prices(PRICE_CACHE, tickers=tickers,
                                     force_download=args.force_download)
    from signal_lab.eventvol import vix as vix_mod
    vix = vix_mod.download_vix(VIX_CACHE, force_download=args.force_download)
    earnings = frame_mod.load_earnings(
        [universe_mod.yfinance_ticker(t) for t in tickers if t != "SPY"],
        force_download=args.force_download)
    events = frame_mod.build_tailq_frame(earnings, prices)

    res = run_dev_eval(prices, events, vix, pooled_cache="data/tailq_pooled.parquet")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"[tailq] verdict: {res['verdict']}")
    print(f"[tailq] results -> {out}")


if __name__ == "__main__":
    main()
