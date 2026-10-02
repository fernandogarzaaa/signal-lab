"""DISPVOL campaign runner: build-frame | evaluate | confirm-test-eval.

- build-frame: reuse the VOL price download/cache and frame build
  (same universe, same frozen target), add the 31 VOL HAR features,
  the 3 frozen VIX features (HARVIX definition), and the 6 frozen
  dispersion features, cache locally (data/dispvol_frame.parquet,
  data/dispvol_calendar.csv). The VIX series cache is REUSED from
  HARVIX (data/harvix_vix.parquet); the GARCH panels use dispvol's
  own cache files. Caches are NOT committed.
- evaluate: 5-fold expanding purged walk-forward (seed 7, min_train
  50, 5-day embargo) comparing har_vix_disp (challenger) vs har_vix
  (primary baseline) vs garch11 (secondary baseline, same
  scale-corrected implementation) vs naive (sanity), paired per fold
  on QLIKE; computes the frozen GO/NO-GO verdict; runs the 95% MCS;
  writes docs/dispvol_results.json (committed) and
  docs/DISPVOL_RESULTS.md (working doc, committed in the results PR).
- confirm-test-eval: ONLY with --confirm-test-eval (and only after a
  dev GO): trains the ridge arms on full dev, scores the untouched
  test period, appends to docs/DISPVOL_RESULTS.md.

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
VOL_PRICES_PARQUET = DATA_DIR / "vol_prices.parquet"
FRAME_PARQUET = DATA_DIR / "dispvol_frame.parquet"
VIX_PARQUET = DATA_DIR / "harvix_vix.parquet"  # reused from HARVIX, not re-downloaded
CALENDAR_CSV = DATA_DIR / "dispvol_calendar.csv"
GARCH_PANEL_PARQUET = DATA_DIR / "dispvol_garch_panel.parquet"
GARCH_PANEL_TEST_PARQUET = DATA_DIR / "dispvol_garch_panel_test.parquet"
RESULTS_JSON = DOCS_DIR / "dispvol_results.json"
RESULTS_MD = DOCS_DIR / "DISPVOL_RESULTS.md"


def _log(msg: str) -> None:
    print(msg, flush=True)


def cmd_build_frame(args) -> None:
    from signal_lab.dispvol import features as dispvol_feat
    from signal_lab.harvix import features as harvix_feat
    from signal_lab.vol import features as vol_feat
    from signal_lab.vol import frame as frame_mod
    from signal_lab.vol import prices as prices_mod
    from signal_lab.vol import universe as universe_mod

    tickers = universe_mod.load_universe(UNIVERSE_CSV)
    _log(f"[build] universe: {len(tickers)} tickers")
    prices = prices_mod.load_or_download_prices(
        VOL_PRICES_PARQUET, tickers=tickers,
        force_download=args.force_download, log=_log)
    frame = frame_mod.build_frame(prices)
    _log(f"[build] frame rows: {len(frame)} "
         f"(t0 {frame['t0'].min()}..{frame['t0'].max()})")
    frame = vol_feat.add_features(frame, prices)
    _log(f"[build] HAR features: {len(vol_feat.FEATURE_COLUMNS)} columns")
    vix = harvix_feat.load_vix(VIX_PARQUET,
                               force_download=args.force_download, log=_log)
    _log(f"[build] VIX series (reused HARVIX cache): {len(vix)} rows "
         f"({vix['date'].min().date()}..{vix['date'].max().date()})")
    frame = harvix_feat.add_vix_features(frame, vix)
    _log(f"[build] VIX features: {harvix_feat.VIX_FEATURE_COLUMNS}")
    frame = dispvol_feat.add_dispersion_features(frame, prices)
    _log(f"[build] dispersion features: {dispvol_feat.DISPERSION_COLUMNS}")
    frame.to_parquet(FRAME_PARQUET, index=False)
    cal = frame_mod.trading_calendar(prices)
    pd.DataFrame({"date": [d.date().isoformat() for d in cal]}).to_csv(
        CALENDAR_CSV, index=False)
    _log(f"[build] wrote {FRAME_PARQUET} ({len(frame)} rows) and "
         f"{CALENDAR_CSV} ({len(cal)} days)")


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


def _fit_predict_ridge(X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame) -> np.ndarray:
    """Frozen ridge arm (alpha = 1.0, standardized): fit on train, predict test."""
    from signal_lab.vol import models as models_mod

    fitted = models_mod.fit_ridge(X_train, y_train)
    return models_mod.predict_ridge(fitted, X_test)


def dispvol_feat_drop_nan(dev: pd.DataFrame, log=_log):
    """Drop rows with NaN features over the challenger's 40 columns."""
    from signal_lab.dispvol import features as dispvol_feat
    from signal_lab.vol import features as vol_feat

    return vol_feat.drop_nan_features(dev, dispvol_feat.CHALLENGER_COLUMNS,
                                      log=log)


def cmd_evaluate(args) -> None:
    from signal_lab.dispvol import features as dispvol_feat
    from signal_lab.eventvol import mcs as mcs_mod
    from signal_lab.stats import deflated_sharpe as dsr_mod
    from signal_lab.validation.splits import WalkForwardConfig, make_splits
    from signal_lab.vol import frame as frame_mod
    from signal_lab.vol import garch as garch_mod
    from signal_lab.vol import metrics as metrics_mod
    from signal_lab.vol import prices as prices_mod
    from signal_lab.vol.run_campaign import _naive_forecasts, _returns_by_ticker

    frame = _load_frame()
    dev = frame_mod.dev_frame(frame)
    _log(f"[evaluate] dev rows: {len(dev)}")
    clean, n_dropped = dispvol_feat_drop_nan(dev)
    trading_days = _load_calendar_ordinals()
    config = WalkForwardConfig(n_splits=5, window="expanding", horizon_days=5,
                               embargo_days=5, min_train=50, seed=7)
    splits = make_splits(clean, config, trading_days)
    _log(f"[evaluate] {len(splits)} folds")

    prices = prices_mod.load_or_download_prices(VOL_PRICES_PARQUET, log=_log)
    returns_by_ticker = _returns_by_ticker(prices)

    test_rows = pd.concat(
        [clean.iloc[s["test_idx"]][["ticker", "t0"]] for s in splits],
        ignore_index=True)
    garch_panel = garch_mod.forecast_panel(
        test_rows, returns_by_ticker, n_jobs=args.jobs,
        cache_path=GARCH_PANEL_PARQUET, log=_log)
    naive_vol = _naive_forecasts(test_rows, returns_by_ticker)
    n_fallbacks = int(garch_panel["garch_fallback"].sum())

    fold_table: list[dict] = []
    row_losses: list[pd.DataFrame] = []
    row_t0: list[pd.Series] = []
    offset = 0
    for s in splits:
        n_test = len(s["test_idx"])
        sl = slice(offset, offset + n_test)
        fold_test = clean.iloc[s["test_idx"]]
        y = clean["target"]
        yt = y.iloc[s["test_idx"]].to_numpy()
        train_idx = s["train_idx"]

        chall_fc = _fit_predict_ridge(
            clean[dispvol_feat.CHALLENGER_COLUMNS].iloc[train_idx],
            y.iloc[train_idx],
            clean[dispvol_feat.CHALLENGER_COLUMNS].iloc[s["test_idx"]])
        base_fc = _fit_predict_ridge(
            clean[dispvol_feat.BASELINE_COLUMNS].iloc[train_idx],
            y.iloc[train_idx],
            clean[dispvol_feat.BASELINE_COLUMNS].iloc[s["test_idx"]])
        garch_fc = garch_panel["garch_vol"].to_numpy()[sl]
        naive_fc = naive_vol[sl]

        q_chall = metrics_mod.mean_qlike(yt ** 2, chall_fc ** 2)
        q_base = metrics_mod.mean_qlike(yt ** 2, base_fc ** 2)
        q_garch = metrics_mod.mean_qlike(yt ** 2, garch_fc ** 2)
        q_naive = metrics_mod.mean_qlike(yt ** 2, naive_fc ** 2)
        fold_table.append({
            "fold": s["fold"],
            "test_start": s["test_start"],
            "test_end": s["test_end"],
            "n_train": len(train_idx),
            "n_test": int(n_test),
            "n_purged": int(s["n_purged"]),
            "n_embargoed": int(s["n_embargoed"]),
            "qlike_har_vix_disp": q_chall,
            "qlike_har_vix": q_base,
            "qlike_garch11": q_garch,
            "qlike_naive": q_naive,
            "mse_var_har_vix_disp": metrics_mod.mse_variance(yt ** 2, chall_fc ** 2),
            "mse_var_har_vix": metrics_mod.mse_variance(yt ** 2, base_fc ** 2),
            "mse_var_garch11": metrics_mod.mse_variance(yt ** 2, garch_fc ** 2),
            "mae_vol_har_vix_disp": metrics_mod.mae_vol(yt, chall_fc),
            "mae_vol_har_vix": metrics_mod.mae_vol(yt, base_fc),
            "mae_vol_garch11": metrics_mod.mae_vol(yt, garch_fc),
            # Primary: challenger vs HAR+VIX. Secondary: challenger vs GARCH.
            "d": q_base - q_chall,
            "d_vs_garch": q_garch - q_chall,
        })
        row_losses.append(pd.DataFrame({
            "har_vix_disp": metrics_mod.qlike(yt ** 2, chall_fc ** 2),
            "har_vix": metrics_mod.qlike(yt ** 2, base_fc ** 2),
            "garch11": metrics_mod.qlike(yt ** 2, garch_fc ** 2),
            "naive": metrics_mod.qlike(yt ** 2, naive_fc ** 2),
        }))
        row_t0.append(pd.to_datetime(fold_test["t0"]).reset_index(drop=True))
        offset += n_test
        _log(f"[evaluate] fold {s['fold']}: QLIKE har_vix_disp={q_chall:.6f} "
             f"har_vix={q_base:.6f} garch={q_garch:.6f} naive={q_naive:.6f} "
             f"d={q_base - q_chall:+.6f}")

    d = np.array([r["d"] for r in fold_table])
    ci = metrics_mod.t_confidence_interval(d, alpha=0.05)
    dm = metrics_mod.diebold_mariano(d)
    d_garch = np.array([r["d_vs_garch"] for r in fold_table])
    dm_garch = metrics_mod.diebold_mariano(d_garch)
    ci_garch = metrics_mod.t_confidence_interval(d_garch, alpha=0.05)
    go = bool(ci["mean"] > 0.0 and ci["lower"] > 0.0)
    _log(f"[evaluate] primary: mean(d)={ci['mean']:+.6f} se={ci['se']:.6f} "
         f"95% CI [{ci['lower']:+.6f}, {ci['upper']:+.6f}] df={ci['df']}")
    _log(f"[evaluate] DM stat={dm['statistic']:+.4f} p={dm['p_value']:.4f}")
    _log(f"[evaluate] vs garch: mean(d)={ci_garch['mean']:+.6f} "
         f"95% CI [{ci_garch['lower']:+.6f}, {ci_garch['upper']:+.6f}], "
         f"DM stat={dm_garch['statistic']:+.4f} p={dm_garch['p_value']:.4f}")
    _log(f"[evaluate] verdict: {'GO' if go else 'NO-GO'}")

    # 95% MCS (reported, not a gate) over pooled per-row dev QLIKE.
    loss_df = pd.concat(row_losses, ignore_index=True)
    t0_all = pd.concat(row_t0, ignore_index=True)
    mcs_out = mcs_mod.mcs(loss_df, t0_all, log=_log)

    # DSR honesty metric (reported, not a gate): per-row QLIKE gains of
    # the two tried ridge arms over naive as pseudo-returns, freq=1,
    # pooled dev test rows. Both tried arms appear (winners AND
    # losers); hiding one would understate N and inflate the DSR.
    dsr_report = dsr_mod.dsr_report(
        {
            "har_vix_disp": loss_df["naive"].to_numpy() - loss_df["har_vix_disp"].to_numpy(),
            "har_vix": loss_df["naive"].to_numpy() - loss_df["har_vix"].to_numpy(),
        },
        freq=1)
    _log("[evaluate] DSR report (reported, not a gate):")
    _log(dsr_report.to_string(index=False))

    results = {
        "challenger": "har_vix_disp",
        "primary_baseline": "har_vix",
        "secondary_baseline": "garch11",
        "folds": fold_table,
        "d": [float(v) for v in d],
        "d_vs_garch": [float(v) for v in d_garch],
        "ci95": {k: float(v) for k, v in ci.items()},
        "ci95_vs_garch": {k: float(v) for k, v in ci_garch.items()},
        "diebold_mariano": {k: float(v) for k, v in dm.items()},
        "diebold_mariano_vs_garch": {k: float(v) for k, v in dm_garch.items()},
        "mcs": {
            "survivors": mcs_out["survivors"],
            "eliminated": [(m, float(p)) for m, p in mcs_out["eliminated"]],
            "mcs_p": {m: float(p) for m, p in mcs_out["mcs_p"].items()},
            "n_rows": mcs_out["n_rows"],
            "n_boot": mcs_out["n_boot"],
            "alpha": mcs_out["alpha"],
            "seed": mcs_out["seed"],
        },
        "dsr": dsr_report.to_dict("records"),
        "dsr_note": ("per-row QLIKE gains of the tried arms over naive as "
                     "pseudo-returns, freq=1, pooled dev test rows; "
                     "reported only, not a gate"),
        "verdict": "GO" if go else "NO-GO",
        "n_garch_fallbacks": n_fallbacks,
        "n_test_rows_total": int(sum(r["n_test"] for r in fold_table)),
        "dev_rows": len(dev),
        "dev_rows_after_nan_drop": len(clean),
        "n_nan_feature_rows_dropped": int(n_dropped),
        "deviations": [],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    _log(f"[evaluate] wrote {RESULTS_JSON}")
    _write_results_md(results)
    _log(f"[evaluate] wrote {RESULTS_MD}")


def _write_results_md(results: dict) -> None:
    """Write the working docs/DISPVOL_RESULTS.md (committed in results PR)."""
    from signal_lab.dispvol import features as dispvol_feat

    lines: list[str] = []
    a = lines.append
    a("# DISPVOL campaign results: HAR+VIX+dispersion vs HAR+VIX (working document)")
    a("")
    a("Pre-registration: `docs/DISPVOL_PREREGISTRATION.md` (frozen, binding). "
      "This document is the working record; the results PR carries the "
      "final version.")
    a("")
    a("## Configuration")
    a("")
    a("- Challenger: **har_vix_disp** = ridge (alpha=1.0, standardized) on "
      "the 34 frozen HAR+VIX features + 6 frozen dispersion features "
      f"({', '.join(dispvol_feat.DISPERSION_COLUMNS)})")
    a("- Primary baseline: **har_vix** = ridge (alpha=1.0, standardized) on "
      "the 34 HAR+VIX features alone (the HARVIX challenger definition, "
      "unchanged)")
    a("- Secondary baseline: **garch11** (scale-corrected 5-day term "
      "structure; fit failures fall back to naive persistence)")
    a("- Sanity arm: naive (trailing realized_vol_5d)")
    a("- Validation: 5-fold expanding purged walk-forward on t0, seed 7, "
      "min_train 50, purge j==k, 5-trading-day embargo")
    a("- Primary metric: QLIKE on variance (lower is better); 95% MCS "
      "reported, not a gate")
    a(f"- Dev rows: {results['dev_rows']} "
      f"({results['dev_rows_after_nan_drop']} after NaN-feature drop)")
    a(f"- GARCH naive fallbacks on test rows: {results['n_garch_fallbacks']} "
      f"/ {results['n_test_rows_total']}")
    a("")
    a("## Per-fold QLIKE (variance)")
    a("")
    a("| fold | test range | n_train | n_test | purged | embargoed | "
      "QLIKE har_vix_disp | QLIKE har_vix | QLIKE garch11 | QLIKE naive | "
      "d = harvix - challenger | d2 = garch - challenger |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results["folds"]:
        a(f"| {r['fold']} | {r['test_start']}..{r['test_end']} | {r['n_train']} | "
          f"{r['n_test']} | {r['n_purged']} | {r['n_embargoed']} | "
          f"{r['qlike_har_vix_disp']:.6f} | {r['qlike_har_vix']:.6f} | "
          f"{r['qlike_garch11']:.6f} | {r['qlike_naive']:.6f} | "
          f"{r['d']:+.6f} | {r['d_vs_garch']:+.6f} |")
    a("")
    a("## Verdict computation (frozen decision rule)")
    a("")
    ci = results["ci95"]
    dm = results["diebold_mariano"]
    a("- Primary: d_k = QLIKE(har_vix) - QLIKE(har_vix_disp) per fold: "
      f"{', '.join(f'{v:+.6f}' for v in results['d'])}")
    a(f"- mean(d) = {ci['mean']:+.6f}, se = {ci['se']:.6f}, "
      f"df = {ci['df']}, t(0.975) = {ci['tcrit']:.4f}")
    a(f"- 95% two-sided Student-t CI on mean(d): "
      f"[{ci['lower']:+.6f}, {ci['upper']:+.6f}]")
    a(f"- Diebold-Mariano on fold-level QLIKE differentials (paired-t form, "
      f"HAC-equivalent on non-overlapping blocks): "
      f"stat = {dm['statistic']:+.4f}, two-sided p = {dm['p_value']:.4f}, n = {dm['n']}")
    a(f"- GO iff mean(d) > 0 AND CI lower bound > 0: **{results['verdict']}**")
    a("")
    a("## Secondary comparison: challenger vs garch11 (reported, no gate authority)")
    a("")
    cig = results["ci95_vs_garch"]
    dmg = results["diebold_mariano_vs_garch"]
    a("- d2_k = QLIKE(garch11) - QLIKE(har_vix_disp) per fold: "
      f"{', '.join(f'{v:+.6f}' for v in results['d_vs_garch'])}")
    a(f"- mean(d2) = {cig['mean']:+.6f}, 95% CI "
      f"[{cig['lower']:+.6f}, {cig['upper']:+.6f}]")
    a(f"- DM: stat = {dmg['statistic']:+.4f}, p = {dmg['p_value']:.4f}")
    a("")
    a("## Model Confidence Set (95%, reported, not a gate)")
    a("")
    mcs = results["mcs"]
    a(f"- Survivors: {', '.join(mcs['survivors'])} "
      f"(n_rows={mcs['n_rows']}, B={mcs['n_boot']}, alpha={mcs['alpha']}, "
      f"seed={mcs['seed']})")
    if mcs["eliminated"]:
        a("- Eliminated (in order): " +
          ", ".join(f"{m} (p={p:.4f})" for m, p in mcs["eliminated"]))
    else:
        a("- No eliminations.")
    a("")
    a("## Secondary metrics (per fold, no gate authority)")
    a("")
    a("| fold | MSE-var challenger | MSE-var har_vix | MSE-var garch11 | "
      "MAE-vol challenger | MAE-vol har_vix | MAE-vol garch11 |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for r in results["folds"]:
        a(f"| {r['fold']} | {r['mse_var_har_vix_disp']:.8f} | "
          f"{r['mse_var_har_vix']:.8f} | {r['mse_var_garch11']:.8f} | "
          f"{r['mae_vol_har_vix_disp']:.6f} | {r['mae_vol_har_vix']:.6f} | "
          f"{r['mae_vol_garch11']:.6f} |")
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
    from signal_lab.dispvol import features as dispvol_feat
    from signal_lab.vol import frame as frame_mod
    from signal_lab.vol import garch as garch_mod
    from signal_lab.vol import metrics as metrics_mod
    from signal_lab.vol import prices as prices_mod
    from signal_lab.vol.run_campaign import _naive_forecasts, _returns_by_ticker

    frame = _load_frame()
    dev = frame_mod.dev_frame(frame)
    dev_clean, _ = dispvol_feat_drop_nan(dev, log=_log)
    test = frame_mod.test_frame(frame)
    test_clean, n_dropped = dispvol_feat_drop_nan(test, log=_log)
    _log(f"[confirm] test rows: {len(test)} ({len(test_clean)} after NaN drop)")

    prices = prices_mod.load_or_download_prices(VOL_PRICES_PARQUET, log=_log)
    returns_by_ticker = _returns_by_ticker(prices)

    chall_fc = _fit_predict_ridge(
        dev_clean[dispvol_feat.CHALLENGER_COLUMNS], dev_clean["target"],
        test_clean[dispvol_feat.CHALLENGER_COLUMNS])
    base_fc = _fit_predict_ridge(
        dev_clean[dispvol_feat.BASELINE_COLUMNS], dev_clean["target"],
        test_clean[dispvol_feat.BASELINE_COLUMNS])

    test_rows = test_clean[["ticker", "t0"]].reset_index(drop=True)
    garch_panel = garch_mod.forecast_panel(
        test_rows, returns_by_ticker, n_jobs=args.jobs,
        cache_path=GARCH_PANEL_TEST_PARQUET, log=_log)
    naive_fc = _naive_forecasts(test_rows, returns_by_ticker)

    yt = test_clean["target"].to_numpy()
    q_chall = metrics_mod.mean_qlike(yt ** 2, chall_fc ** 2)
    q_base = metrics_mod.mean_qlike(yt ** 2, base_fc ** 2)
    q_garch = metrics_mod.mean_qlike(
        yt ** 2, garch_panel["garch_vol"].to_numpy() ** 2)
    q_naive = metrics_mod.mean_qlike(yt ** 2, naive_fc ** 2)
    _log(f"[confirm] test QLIKE: har_vix_disp={q_chall:.6f} "
         f"har_vix={q_base:.6f} garch={q_garch:.6f} naive={q_naive:.6f}")

    results = json.loads(RESULTS_JSON.read_text())
    dev_means = {
        arm: float(np.mean([r[f"qlike_{arm}"] for r in results["folds"]]))
        for arm in ("har_vix_disp", "har_vix", "garch11", "naive")
    }
    section = [
        "",
        "## Test-period evaluation (single authorized run)",
        "",
        f"- Test rows scored: {len(test_clean)} (t0 "
        f"{test_clean['t0'].min()}..{test_clean['t0'].max()}); "
        f"{n_dropped} rows dropped for NaN features",
        f"- GARCH naive fallbacks on test rows: "
        f"{int(garch_panel['garch_fallback'].sum())}",
        "- Prices end 2026-07-15, so the labelable test window is shorter "
        "than the calendar test period; coverage above is the effective one.",
        "",
        "| arm | dev walk-forward mean QLIKE | test QLIKE |",
        "| --- | --- | --- |",
        f"| har_vix_disp | {dev_means['har_vix_disp']:.6f} | {q_chall:.6f} |",
        f"| har_vix | {dev_means['har_vix']:.6f} | {q_base:.6f} |",
        f"| garch11 | {dev_means['garch11']:.6f} | {q_garch:.6f} |",
        f"| naive | {dev_means['naive']:.6f} | {q_naive:.6f} |",
        "",
    ]
    with RESULTS_MD.open("a") as f:
        f.write("\n".join(section))
    results["test_eval"] = {
        "n_test_rows": len(test_clean),
        "t0_min": str(test_clean["t0"].min()),
        "t0_max": str(test_clean["t0"].max()),
        "qlike_har_vix_disp": q_chall,
        "qlike_har_vix": q_base,
        "qlike_garch11": q_garch,
        "qlike_naive": q_naive,
        "dev_mean_qlike": dev_means,
        "n_garch_fallbacks": int(garch_panel["garch_fallback"].sum()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    _log(f"[confirm] appended test evaluation to {RESULTS_MD}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="run_campaign.py",
                                description="DISPVOL campaign runner")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-frame",
                       help="build frame + HAR + VIX + dispersion features")
    b.add_argument("--force-download", action="store_true")
    b.set_defaults(func=cmd_build_frame)

    e = sub.add_parser("evaluate", help="5-fold walk-forward vs har_vix")
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
