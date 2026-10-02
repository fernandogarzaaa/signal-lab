"""EVENTVOL campaign runner: build-frame | evaluate | confirm-test-eval.

- build-frame: download/cache prices (frozen universe_vol.csv window),
  VIX history, and earnings calendars; run quality gates; build the
  earnings-event frame with frozen targets + features; cache locally
  (data/eventvol_frame.parquet, data/eventvol_calendar.csv,
  data/eventvol_vix.parquet, data/eventvol_earnings.parquet). Caches are
  NOT committed.
- evaluate: 5-fold expanding purged walk-forward (seed 7, min_train 50,
  5-day embargo) comparing the challenger (LightGBM on HAR+VIX+event
  features) against iv_proxy (primary baseline), garch11, har_vix, and
  naive, paired per fold on QLIKE; computes the frozen GO/NO-GO verdict
  (paired-t CI on d = QLIKE(iv_proxy) - QLIKE(challenger) plus the 95%
  Model Confidence Set); reports DSR as a honesty metric (not a gate);
  writes docs/eventvol_results.json and docs/EVENTVOL_RESULTS.md
  (working doc, committed in the results PR).
- confirm-test-eval: ONLY with --confirm-test-eval (and only after a dev
  GO): trains on full dev, scores the untouched test period once,
  appends to docs/EVENTVOL_RESULTS.md.

Every command re-enforces check_no_confirmation on t0; test and
confirmation periods are never touched during development.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
DOCS_DIR = REPO_ROOT / "docs"

UNIVERSE_CSV = DATA_DIR / "universe_vol.csv"
PRICES_PARQUET = DATA_DIR / "vol_prices.parquet"
VIX_PARQUET = DATA_DIR / "eventvol_vix.parquet"
EARNINGS_PARQUET = DATA_DIR / "eventvol_earnings.parquet"
FRAME_PARQUET = DATA_DIR / "eventvol_frame.parquet"
CALENDAR_CSV = DATA_DIR / "eventvol_calendar.csv"
GARCH_PANEL_PARQUET = DATA_DIR / "eventvol_garch_panel.parquet"
RESULTS_JSON = DOCS_DIR / "eventvol_results.json"
RESULTS_MD = DOCS_DIR / "EVENTVOL_RESULTS.md"

ARM_NAMES = ["challenger", "iv_proxy", "garch11", "har_vix", "naive"]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _returns_by_ticker(prices: pd.DataFrame) -> dict[str, pd.Series]:
    """ticker -> daily log-return Series indexed by date (adj closes)."""
    out: dict[str, pd.Series] = {}
    for t, grp in prices.sort_values(["ticker", "date"]).groupby("ticker"):
        g = grp.set_index("date").sort_index()
        logc = np.log(g["adj_close"].to_numpy(dtype=float))
        r = np.empty(len(g))
        r[0] = np.nan
        r[1:] = logc[1:] - logc[:-1]
        out[str(t)] = pd.Series(r, index=g.index)
    return out


def _naive_forecasts(rows: pd.DataFrame,
                     returns_by_ticker: dict[str, pd.Series]) -> np.ndarray:
    """Trailing realized_vol_5d persistence forecast per row."""
    from signal_lab.vol import garch as garch_mod

    vols = np.empty(len(rows))
    for i, (t, t0) in enumerate(zip(rows["ticker"].astype(str),
                                    pd.to_datetime(rows["t0"]).dt.date)):
        rs = returns_by_ticker[t].sort_index()
        dates = np.array([d.toordinal() for d in rs.index.date])
        pos = int(np.searchsorted(dates, t0.toordinal(), side="left"))
        vols[i] = garch_mod.naive_forecast(rs.to_numpy(dtype=float), pos)
    if int(np.isnan(vols).sum()):
        raise ValueError("[eventvol] NaN naive forecasts; refusing to score")
    return vols


def _load_frame() -> pd.DataFrame:
    if not FRAME_PARQUET.exists():
        raise FileNotFoundError(
            f"{FRAME_PARQUET} missing; run build-frame first")
    frame = pd.read_parquet(FRAME_PARQUET)
    frame["t0"] = pd.to_datetime(frame["t0"]).dt.date
    frame["t1"] = pd.to_datetime(frame["t1"]).dt.date
    return frame


def _load_calendar_ordinals() -> np.ndarray:
    if not CALENDAR_CSV.exists():
        raise FileNotFoundError(
            f"{CALENDAR_CSV} missing; run build-frame first")
    cal = pd.read_csv(CALENDAR_CSV, parse_dates=["date"])
    return np.array(sorted(d.date().toordinal() for d in cal["date"]),
                     dtype=np.int64)


def cmd_build_frame(args) -> None:
    from signal_lab.eventvol import earnings as earn_mod
    from signal_lab.eventvol import features as feat_mod
    from signal_lab.eventvol import vix as vix_mod
    from signal_lab.vol import frame as vol_frame_mod
    from signal_lab.vol import prices as prices_mod
    from signal_lab.vol import universe as universe_mod

    tickers = universe_mod.load_universe(UNIVERSE_CSV)
    _log(f"[build] universe: {len(tickers)} tickers")
    prices = prices_mod.load_or_download_prices(
        PRICES_PARQUET, tickers=tickers,
        force_download=args.force_download, log=_log)
    vix = vix_mod.download_vix(
        VIX_PARQUET, force_download=args.force_download, log=_log)
    earnings = earn_mod.download_earnings(
        tickers, EARNINGS_PARQUET,
        force_download=args.force_download, log=_log)
    frame = earn_mod.build_event_frame(earnings, prices, log=_log)
    frame = feat_mod.add_features(frame, prices, vix, log=_log)
    frame.to_parquet(FRAME_PARQUET, index=False)
    cal = vol_frame_mod.trading_calendar(prices)
    pd.DataFrame({"date": [d.date().isoformat() for d in cal]}).to_csv(
        CALENDAR_CSV, index=False)
    _log(f"[build] wrote {FRAME_PARQUET} ({len(frame)} rows) and "
         f"{CALENDAR_CSV} ({len(cal)} days)")


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                 feature_cols: list[str], arm: str) -> np.ndarray:
    from signal_lab.vol import models as models_mod

    X = train[feature_cols]
    y = train["target"]
    fitted = models_mod.fit_arm(arm, X, y)
    return models_mod.predict_arm(arm, fitted, test[feature_cols])


def cmd_evaluate(args) -> None:
    from signal_lab.eventvol import earnings as earn_mod
    from signal_lab.eventvol import features as feat_mod
    from signal_lab.eventvol import iv_proxy as iv_mod
    from signal_lab.eventvol import mcs as mcs_mod
    from signal_lab.stats import deflated_sharpe as dsr_mod
    from signal_lab.validation.splits import WalkForwardConfig, make_splits
    from signal_lab.vol import garch as garch_mod
    from signal_lab.vol import metrics as metrics_mod
    from signal_lab.vol import prices as prices_mod

    frame = _load_frame()
    dev = earn_mod.dev_frame(frame)
    _log(f"[evaluate] dev event rows: {len(dev)}")
    clean, n_dropped = feat_mod.drop_nan_features(dev, log=_log)
    trading_days = _load_calendar_ordinals()
    config = WalkForwardConfig(n_splits=5, window="expanding", horizon_days=5,
                               embargo_days=5, min_train=50, seed=7)
    splits = make_splits(clean, config, trading_days)
    _log(f"[evaluate] {len(splits)} folds")

    prices = prices_mod.load_or_download_prices(PRICES_PARQUET, log=_log)
    returns_by_ticker = _returns_by_ticker(prices)
    vix = pd.read_parquet(VIX_PARQUET)
    vix["date"] = pd.to_datetime(vix["date"])

    test_rows = pd.concat(
        [clean.iloc[s["test_idx"]][["ticker", "t0"]] for s in splits],
        ignore_index=True)
    garch_panel = garch_mod.forecast_panel(
        test_rows, returns_by_ticker, n_jobs=args.jobs,
        cache_path=GARCH_PANEL_PARQUET, log=_log)
    naive_vol = _naive_forecasts(test_rows, returns_by_ticker)
    iv_fc, iv_counts = iv_mod.forecast_iv_proxy(test_rows, prices, vix,
                                                log=_log)
    n_garch_fallbacks = int(garch_panel["garch_fallback"].sum())

    fold_table: list[dict] = []
    per_row_losses: list[pd.DataFrame] = []
    offset = 0
    for s in splits:
        n_test = len(s["test_idx"])
        sl = slice(offset, offset + n_test)
        fold_test = clean.iloc[s["test_idx"]]
        fold_train = clean.iloc[s["train_idx"]]
        yt = fold_test["target"].to_numpy()
        champ_fc = _fit_predict(fold_train, fold_test,
                                feat_mod.CHALLENGER_FEATURES, "lightgbm")
        harvix_fc = _fit_predict(fold_train, fold_test,
                                 feat_mod.HARVIX_FEATURES, "ridge")
        fc = {
            "challenger": champ_fc,
            "iv_proxy": iv_fc[sl],
            "garch11": garch_panel["garch_vol"].to_numpy()[sl],
            "har_vix": harvix_fc,
            "naive": naive_vol[sl],
        }
        q = {name: metrics_mod.mean_qlike(yt ** 2, fc[name] ** 2)
             for name in ARM_NAMES}
        fold_table.append({
            "fold": s["fold"],
            "test_start": s["test_start"],
            "test_end": s["test_end"],
            "n_train": int(len(s["train_idx"])),
            "n_test": int(n_test),
            "n_purged": int(s["n_purged"]),
            "n_embargoed": int(s["n_embargoed"]),
            **{f"qlike_{name}": q[name] for name in ARM_NAMES},
            "mse_var_challenger": metrics_mod.mse_variance(
                yt ** 2, fc["challenger"] ** 2),
            "mse_var_iv_proxy": metrics_mod.mse_variance(
                yt ** 2, fc["iv_proxy"] ** 2),
            "mae_vol_challenger": metrics_mod.mae_vol(yt, fc["challenger"]),
            "mae_vol_iv_proxy": metrics_mod.mae_vol(yt, fc["iv_proxy"]),
            "d": q["iv_proxy"] - q["challenger"],
        })
        per_row_losses.append(pd.DataFrame(
            {name: metrics_mod.qlike(yt ** 2, fc[name] ** 2)
             for name in ARM_NAMES}))
        offset += n_test
        _log(f"[evaluate] fold {s['fold']}: " +
             " ".join(f"{name}={q[name]:.6f}" for name in ARM_NAMES) +
             f" d={q['iv_proxy'] - q['challenger']:+.6f}")

    d = np.array([r["d"] for r in fold_table])
    ci = metrics_mod.t_confidence_interval(d, alpha=0.05)
    dm = metrics_mod.diebold_mariano(d)

    loss_df = pd.concat(per_row_losses, ignore_index=True)
    t0_all = pd.concat(
        [clean.iloc[s["test_idx"]]["t0"] for s in splits], ignore_index=True)
    mcs_res = mcs_mod.mcs(loss_df, t0_all, log=_log)
    challenger_in_mcs = "challenger" in mcs_res["survivors"]

    go = bool(ci["mean"] > 0.0 and ci["lower"] > 0.0 and challenger_in_mcs)
    _log(f"[evaluate] mean(d)={ci['mean']:+.6f} se={ci['se']:.6f} "
         f"95% CI [{ci['lower']:+.6f}, {ci['upper']:+.6f}] df={ci['df']}")
    _log(f"[evaluate] DM stat={dm['statistic']:+.4f} p={dm['p_value']:.4f}")
    _log(f"[evaluate] MCS survivors: {mcs_res['survivors']}")
    _log(f"[evaluate] verdict: {'GO' if go else 'NO-GO'}")

    # DSR honesty metric (reported, not a gate): per-row QLIKE gains over
    # naive as pseudo-returns, freq=1.
    naive_q = loss_df["naive"].to_numpy()
    variant_returns = {
        name: (naive_q - loss_df[name].to_numpy()) for name in ARM_NAMES
        if name != "naive"
    }
    dsr_report = dsr_mod.dsr_report(variant_returns, freq=1)
    _log("[evaluate] DSR report (reported, not a gate):")
    _log(dsr_report.to_string(index=False))

    results = {
        "folds": fold_table,
        "d": [float(v) for v in d],
        "ci95": {k: float(v) for k, v in ci.items()},
        "diebold_mariano": {k: float(v) for k, v in dm.items()},
        "mcs": {
            "survivors": mcs_res["survivors"],
            "eliminated": mcs_res["eliminated"],
            "mcs_p": mcs_res["mcs_p"],
            "n_rows": mcs_res["n_rows"],
            "alpha": mcs_res["alpha"],
            "n_boot": mcs_res["n_boot"],
        },
        "challenger_in_mcs": challenger_in_mcs,
        "verdict": "GO" if go else "NO-GO",
        "n_garch_fallbacks": n_garch_fallbacks,
        "iv_proxy_fallbacks": iv_counts,
        "n_test_rows_total": int(sum(r["n_test"] for r in fold_table)),
        "dev_rows": int(len(dev)),
        "dev_rows_after_nan_drop": int(len(clean)),
        "n_nan_feature_rows_dropped": int(n_dropped),
        "dsr": dsr_report.to_dict("records"),
        "dsr_note": ("per-row QLIKE gains over naive as pseudo-returns, "
                     "freq=1; reported only, not a gate"),
        "deviations": [],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    _log(f"[evaluate] wrote {RESULTS_JSON}")
    _write_results_md(results)
    _log(f"[evaluate] wrote {RESULTS_MD}")


def _write_results_md(results: dict) -> None:
    """Write the working docs/EVENTVOL_RESULTS.md (results PR carries it)."""
    lines: list[str] = []
    a = lines.append
    a("# EVENTVOL campaign results: event-window volatility (working document)")
    a("")
    a("Pre-registration: `docs/EVENTVOL_PREREGISTRATION.md` (frozen, "
      "binding). This document is the working record; the committed results "
      "PR carries the final version.")
    a("")
    a("## Configuration")
    a("")
    a("- Challenger: LightGBM (n_estimators=300, lr=0.05, leaves=31, "
      "min_child=100, feat_frac=0.8, bag_frac=0.8, l2=1.0, seed=7) on "
      "HAR(31) + VIX(4) + event(4) features; hyperparameters frozen, no "
      "selection")
    a("- Primary baseline: iv_proxy = beta(i,t0) x VIX(t0)/100/sqrt(252) "
      "(the market's forward-looking forecast; per-stock historical IV is "
      "not obtainable from yfinance, see pre-registration STEP 0)")
    a("- Secondary baselines: garch11 (scale-corrected 5-day term "
      "structure; fit failures fall back to naive persistence), har_vix "
      "(ridge alpha=1.0 on HAR+VIX features)")
    a("- Sanity arm: naive (trailing realized_vol_5d)")
    a("- Validation: 5-fold expanding purged walk-forward on t0, seed 7, "
      "min_train 50, purge on [t0, t0+5] overlap, 5-trading-day embargo")
    a("- Primary metric: QLIKE on variance (lower is better); second GO "
      "conjunct: challenger survives the 95% Model Confidence Set "
      "(Hansen-Lunde-Nason T_R, day-block stationary bootstrap, B=5000)")
    a(f"- Dev event rows: {results['dev_rows']} "
      f"({results['dev_rows_after_nan_drop']} after NaN-feature drop)")
    a(f"- GARCH naive fallbacks on test rows: {results['n_garch_fallbacks']} "
      f"/ {results['n_test_rows_total']}")
    a(f"- iv_proxy fallbacks: {results['iv_proxy_fallbacks']}")
    a("")
    a("## Per-fold QLIKE (variance)")
    a("")
    a("| fold | test range | n_train | n_test | purged | embargoed | "
      "QLIKE challenger | QLIKE iv_proxy | QLIKE garch11 | QLIKE har_vix | "
      "QLIKE naive | d = iv - champ |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results["folds"]:
        a(f"| {r['fold']} | {r['test_start']}..{r['test_end']} | {r['n_train']} | "
          f"{r['n_test']} | {r['n_purged']} | {r['n_embargoed']} | "
          f"{r['qlike_challenger']:.6f} | {r['qlike_iv_proxy']:.6f} | "
          f"{r['qlike_garch11']:.6f} | {r['qlike_har_vix']:.6f} | "
          f"{r['qlike_naive']:.6f} | {r['d']:+.6f} |")
    a("")
    a("## Verdict computation (frozen decision rule)")
    a("")
    ci = results["ci95"]
    dm = results["diebold_mariano"]
    a(f"- d_k = QLIKE(iv_proxy) - QLIKE(challenger) per fold: "
      f"{', '.join(f'{v:+.6f}' for v in results['d'])}")
    a(f"- mean(d) = {ci['mean']:+.6f}, se = {ci['se']:.6f}, "
      f"df = {ci['df']}, t(0.975) = {ci['tcrit']:.4f}")
    a(f"- 95% two-sided Student-t CI on mean(d): "
      f"[{ci['lower']:+.6f}, {ci['upper']:+.6f}]")
    a(f"- Diebold-Mariano on fold-level QLIKE differentials (paired-t form): "
      f"stat = {dm['statistic']:+.4f}, two-sided p = {dm['p_value']:.4f}, "
      f"n = {dm['n']}")
    a(f"- 95% MCS survivors: {results['mcs']['survivors']}; "
      f"eliminated (in order): {results['mcs']['eliminated']}")
    a(f"- challenger in MCS: {results['challenger_in_mcs']}")
    a(f"- GO iff mean(d) > 0 AND CI lower bound > 0 AND challenger in MCS: "
      f"**{results['verdict']}**")
    a("")
    a("## Secondary metrics (per fold, no gate authority)")
    a("")
    a("| fold | MSE-var challenger | MSE-var iv_proxy | MAE-vol challenger | "
      "MAE-vol iv_proxy |")
    a("| --- | --- | --- | --- | --- |")
    for r in results["folds"]:
        a(f"| {r['fold']} | {r['mse_var_challenger']:.8f} | "
          f"{r['mse_var_iv_proxy']:.8f} | {r['mae_vol_challenger']:.6f} | "
          f"{r['mae_vol_iv_proxy']:.6f} |")
    a("")
    a("## Selection honesty (DSR, reported, not a gate)")
    a("")
    a(results["dsr_note"])
    a("")
    for row in results["dsr"]:
        a(f"- {row['variant']}: Sharpe = {row['sharpe']:+.4f}, "
          f"DSR = {row['dsr']:.4f}, "
          f"likely_false_discovery = {row['likely_false_discovery']}")
    a("")
    a("## Deviations")
    a("")
    if results["deviations"]:
        for dev in results["deviations"]:
            a(f"- {dev}")
    else:
        a("None.")
    a("")
    RESULTS_MD.write_text("\n".join(lines))


def cmd_confirm_test_eval(args) -> None:
    if not args.confirm_test_eval:
        raise SystemExit(
            "refusing: confirm-test-eval requires the explicit "
            "--confirm-test-eval flag (single authorized test evaluation)")
    from signal_lab.eventvol import earnings as earn_mod
    from signal_lab.eventvol import features as feat_mod
    from signal_lab.eventvol import iv_proxy as iv_mod
    from signal_lab.vol import garch as garch_mod
    from signal_lab.vol import metrics as metrics_mod
    from signal_lab.vol import prices as prices_mod

    frame = _load_frame()
    dev = earn_mod.dev_frame(frame)
    dev_clean, _ = feat_mod.drop_nan_features(dev, log=_log)
    test = earn_mod.test_frame(frame)
    test_clean, n_dropped = feat_mod.drop_nan_features(test, log=_log)
    _log(f"[confirm] test event rows: {len(test)} "
         f"({len(test_clean)} after NaN drop)")

    prices = prices_mod.load_or_download_prices(PRICES_PARQUET, log=_log)
    returns_by_ticker = _returns_by_ticker(prices)
    vix = pd.read_parquet(VIX_PARQUET)
    vix["date"] = pd.to_datetime(vix["date"])

    champ_fc = _fit_predict(dev_clean, test_clean,
                            feat_mod.CHALLENGER_FEATURES, "lightgbm")
    harvix_fc = _fit_predict(dev_clean, test_clean,
                             feat_mod.HARVIX_FEATURES, "ridge")
    test_rows = test_clean[["ticker", "t0"]].reset_index(drop=True)
    garch_panel = garch_mod.forecast_panel(
        test_rows, returns_by_ticker, n_jobs=args.jobs,
        cache_path=DATA_DIR / "eventvol_garch_panel_test.parquet", log=_log)
    naive_fc = _naive_forecasts(test_rows, returns_by_ticker)
    iv_fc, iv_counts = iv_mod.forecast_iv_proxy(test_rows, prices, vix,
                                                log=_log)

    yt = test_clean["target"].to_numpy()
    q = {
        "challenger": metrics_mod.mean_qlike(yt ** 2, champ_fc ** 2),
        "iv_proxy": metrics_mod.mean_qlike(yt ** 2, iv_fc ** 2),
        "garch11": metrics_mod.mean_qlike(
            yt ** 2, garch_panel["garch_vol"].to_numpy() ** 2),
        "har_vix": metrics_mod.mean_qlike(yt ** 2, harvix_fc ** 2),
        "naive": metrics_mod.mean_qlike(yt ** 2, naive_fc ** 2),
    }
    _log("[confirm] test QLIKE: " +
         " ".join(f"{name}={q[name]:.6f}" for name in ARM_NAMES))

    results = json.loads(RESULTS_JSON.read_text())
    dev_means = {
        name: float(np.mean([r[f"qlike_{name}"] for r in results["folds"]]))
        for name in ARM_NAMES
    }
    section = [
        "",
        "## Test-period evaluation (single authorized run)",
        "",
        f"- Test event rows scored: {len(test_clean)} (t0 "
        f"{test_clean['t0'].min()}..{test_clean['t0'].max()}); "
        f"{n_dropped} rows dropped for NaN features",
        f"- GARCH naive fallbacks on test rows: "
        f"{int(garch_panel['garch_fallback'].sum())}",
        f"- iv_proxy fallbacks: {iv_counts}",
        "- Prices end 2026-07-15, so the labelable test window is shorter "
        "than the calendar test period; coverage above is the effective one.",
        "",
        "| arm | dev walk-forward mean QLIKE | test QLIKE |",
        "| --- | --- | --- |",
    ]
    for name in ARM_NAMES:
        section.append(
            f"| {name} | {dev_means[name]:.6f} | {q[name]:.6f} |")
    section.append("")
    with RESULTS_MD.open("a") as f:
        f.write("\n".join(section))
    results["test_eval"] = {
        "n_test_rows": int(len(test_clean)),
        "t0_min": str(test_clean["t0"].min()),
        "t0_max": str(test_clean["t0"].max()),
        "qlike": q,
        "dev_mean_qlike": dev_means,
        "n_garch_fallbacks": int(garch_panel["garch_fallback"].sum()),
        "iv_proxy_fallbacks": iv_counts,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    _log(f"[confirm] appended test evaluation to {RESULTS_MD}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="run_eventvol.py",
                                description="EVENTVOL campaign runner")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-frame",
                       help="download data, build event frame + features")
    b.add_argument("--force-download", action="store_true")
    b.set_defaults(func=cmd_build_frame)

    e = sub.add_parser("evaluate", help="5-fold walk-forward vs iv_proxy")
    e.add_argument("--jobs", type=int, default=1)
    e.set_defaults(func=cmd_evaluate)

    c = sub.add_parser("confirm-test-eval",
                       help="single authorized test-period evaluation")
    c.add_argument("--confirm-test-eval", action="store_true",
                   help="explicit confirmation flag (required)")
    c.add_argument("--jobs", type=int, default=1)
    c.set_defaults(func=cmd_confirm_test_eval)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
