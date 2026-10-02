"""CONFMAG campaign runner: build-frame | evaluate | confirm-test-eval.

- build-frame: reuse the VOL price download/cache, build the
  |excess_3d| frame (confmag.features), add the 31 VOL HAR features
  and the 3 frozen VIX features (HARVIX definition), cache locally
  (data/confmag_frame.parquet, data/confmag_calendar.csv). The VIX
  series cache is REUSED from HARVIX (data/harvix_vix.parquet).
  Caches are NOT committed.
- evaluate: 5-fold expanding purged walk-forward (seed 7, min_train
  50, 3-day horizon purge, 3-day embargo) with per-fold
  Conformalized Quantile Regression gating (temporal 20% calibration
  split, LightGBM tau=0.1/0.9, alpha=0.2); the LightGBM L2 point
  forecast is scored on the selected subset vs the per-fold train
  mean (constant) and the GARCH-implied secondary; computes the
  frozen conjunctive GO/NO-GO verdict; writes
  docs/confmag_results.json (committed) and docs/CONFMAG_RESULTS.md
  (working doc, committed in the results PR).
- confirm-test-eval: ONLY with --confirm-test-eval (and only after a
  dev GO): retrains on full dev, scores the untouched test period on
  the gated subset, appends to docs/CONFMAG_RESULTS.md.

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
FRAME_PARQUET = DATA_DIR / "confmag_frame.parquet"
VIX_PARQUET = DATA_DIR / "harvix_vix.parquet"  # reused from HARVIX, not re-downloaded
CALENDAR_CSV = DATA_DIR / "confmag_calendar.csv"
GARCH_PANEL_PARQUET = DATA_DIR / "confmag_garch_panel.parquet"
GARCH_PANEL_TEST_PARQUET = DATA_DIR / "confmag_garch_panel_test.parquet"
RESULTS_JSON = DOCS_DIR / "confmag_results.json"
RESULTS_MD = DOCS_DIR / "CONFMAG_RESULTS.md"

# sqrt(2/pi): E|X| = sigma * sqrt(2/pi) for X ~ N(0, sigma^2).
_SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))
# 3-day cumulative vol from the daily-scale 5-day GARCH forecast.
_GARCH_3D_SCALE = float(np.sqrt(3.0 / 5.0))


def _log(msg: str) -> None:
    print(msg, flush=True)


def cmd_build_frame(args) -> None:
    from signal_lab.confmag import features as confmag_feat
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
    frame = confmag_feat.build_frame(prices)
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


def confmag_feat_drop_nan(dev: pd.DataFrame, log=_log):
    """Drop rows with NaN features over the frozen 34 columns."""
    from signal_lab.confmag import features as confmag_feat
    from signal_lab.vol import features as vol_feat

    return vol_feat.drop_nan_features(dev, confmag_feat.FEATURE_COLUMNS,
                                      log=log)


def _evaluate_fold(clean: pd.DataFrame, train_idx: np.ndarray,
                   test_idx: np.ndarray, garch_vol_test: np.ndarray,
                   log=_log) -> dict:
    """One fold: CQR gating + point forecast, scored on the selected subset."""
    from signal_lab.confmag import cqr as cqr_mod
    from signal_lab.confmag import features as confmag_feat

    cols = confmag_feat.FEATURE_COLUMNS
    y = clean["target"]
    y_train = y.iloc[train_idx].to_numpy(dtype=float)
    y_test = y.iloc[test_idx].to_numpy(dtype=float)
    X_train = clean[cols].iloc[train_idx]
    X_test = clean[cols].iloc[test_idx]

    # Temporal proper-train / calibration split of the train block.
    t0_train = clean["t0"].iloc[train_idx]
    proper_pos, cal_pos = cqr_mod.temporal_proper_cal(t0_train)
    X_proper = X_train.iloc[proper_pos]
    y_proper = y_train[proper_pos]
    X_cal = X_train.iloc[cal_pos]
    y_cal = y_train[cal_pos]

    # Point forecast: LightGBM L2 on the FULL train block.
    point_model = cqr_mod.fit_point_model(X_train, pd.Series(y_train))
    point_fc = cqr_mod.predict_point(point_model, X_test)

    # CQR intervals.
    m_lo, m_hi = cqr_mod.fit_quantile_models(
        X_proper, pd.Series(y_proper))
    cal_out = cqr_mod.conformalize(
        m_lo, m_hi, X_cal, pd.Series(y_cal))
    test_out = cqr_mod.predict_intervals(
        m_lo, m_hi, X_test, cal_out["q_hat"])
    widths = test_out["widths"]
    selected = np.isfinite(widths) & (widths <= cal_out["threshold"])
    n_selected = int(selected.sum())
    coverage = cqr_mod.empirical_coverage(
        y_test, test_out["lo"], test_out["hi"])

    # Baselines on the selected subset.
    const_fc = float(np.mean(y_train))
    garch_implied = (_GARCH_3D_SCALE * garch_vol_test *
                      _SQRT_2_OVER_PI)
    out: dict = {
        "n_test": int(len(test_idx)),
        "n_selected": n_selected,
        "selection_rate": float(n_selected / len(test_idx)),
        "coverage": float(coverage),
        "q_hat": cal_out["q_hat"],
        "threshold": cal_out["threshold"],
        "n_cal": cal_out["n_cal"],
        "const_fc": const_fc,
    }
    if n_selected == 0:
        out.update({"mse_chall": np.nan, "mse_const": np.nan,
                    "mse_garch": np.nan, "d": np.nan})
        return out
    ys = y_test[selected]
    mse_chall = float(np.mean((ys - point_fc[selected]) ** 2))
    mse_const = float(np.mean((ys - const_fc) ** 2))
    mse_garch = float(np.mean((ys - garch_implied[selected]) ** 2))
    out.update({
        "mse_chall": mse_chall,
        "mse_const": mse_const,
        "mse_garch": mse_garch,
        "d": mse_const - mse_chall,
        "d_garch": mse_garch - mse_chall,
        "mae_chall": float(np.mean(np.abs(ys - point_fc[selected]))),
        "mae_const": float(np.mean(np.abs(ys - const_fc))),
        # Per-row squared errors on the selected subset (for MCS/DSR),
        # plus the selected rows' t0s for the MCS day-block bootstrap.
        "se_chall": (ys - point_fc[selected]) ** 2,
        "se_const": (ys - const_fc) ** 2,
        "se_garch": (ys - garch_implied[selected]) ** 2,
        "t0_selected": clean["t0"].iloc[test_idx].to_numpy()[selected],
    })
    return out


def cmd_evaluate(args) -> None:
    from signal_lab.eventvol import mcs as mcs_mod
    from signal_lab.stats import deflated_sharpe as dsr_mod
    from signal_lab.validation.splits import WalkForwardConfig, make_splits
    from signal_lab.vol import frame as frame_mod
    from signal_lab.vol import garch as garch_mod
    from signal_lab.vol import metrics as metrics_mod
    from signal_lab.vol import prices as prices_mod
    from signal_lab.vol.run_campaign import _returns_by_ticker

    frame = _load_frame()
    dev = frame_mod.dev_frame(frame)
    _log(f"[evaluate] dev rows: {len(dev)}")
    clean, n_dropped = confmag_feat_drop_nan(dev)
    trading_days = _load_calendar_ordinals()
    config = WalkForwardConfig(n_splits=5, window="expanding", horizon_days=3,
                               embargo_days=3, min_train=50, seed=7)
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
    garch_vol = garch_panel["garch_vol"].to_numpy(dtype=float)
    n_fallbacks = int(garch_panel["garch_fallback"].sum())

    fold_rows: list[dict] = []
    d_list: list[float] = []
    d_garch_list: list[float] = []
    se_frames: list[pd.DataFrame] = []
    t0_selected_list: list[np.ndarray] = []
    offset = 0
    n_selected_total = 0
    n_test_total = 0
    for s in splits:
        n_test = len(s["test_idx"])
        sl = slice(offset, offset + n_test)
        fr = _evaluate_fold(clean, s["train_idx"], s["test_idx"],
                            garch_vol[sl], log=_log)
        n_selected_total += fr["n_selected"]
        n_test_total += fr["n_test"]
        d_list.append(fr["d"])
        d_garch_list.append(fr["d_garch"])
        if fr["n_selected"] > 0:
            se_frames.append(pd.DataFrame({
                "challenger": fr["se_chall"],
                "constant": fr["se_const"],
                "garch_implied": fr["se_garch"],
            }))
            t0_selected_list.append(fr["t0_selected"])
        fold_rows.append({
            "fold": s["fold"],
            "test_start": s["test_start"],
            "test_end": s["test_end"],
            "n_train": len(s["train_idx"]),
            "n_test": fr["n_test"],
            "n_purged": int(s["n_purged"]),
            "n_embargoed": int(s["n_embargoed"]),
            "n_selected": fr["n_selected"],
            "selection_rate": fr["selection_rate"],
            "coverage": fr["coverage"],
            "q_hat": fr["q_hat"],
            "threshold": fr["threshold"],
            "n_cal": fr["n_cal"],
            "mse_chall": fr["mse_chall"],
            "mse_const": fr["mse_const"],
            "mse_garch": fr["mse_garch"],
            "d": fr["d"],
            "d_garch": fr["d_garch"],
            "mae_chall": fr.get("mae_chall", np.nan),
            "mae_const": fr.get("mae_const", np.nan),
        })
        _log(f"[evaluate] fold {s['fold']}: selected "
             f"{fr['n_selected']}/{fr['n_test']} "
             f"({fr['selection_rate']:.3f}), coverage={fr['coverage']:.3f}, "
             f"MSE chall={fr['mse_chall']:.6f} const={fr['mse_const']:.6f} "
             f"d={fr['d']:+.6f}")
        offset += n_test

    selection_rate = n_selected_total / n_test_total
    d = np.array(d_list, dtype=float)
    valid = np.isfinite(d)
    if not valid.all():
        _log("[evaluate] FAIL-LOUD: a fold selected zero rows; "
             "verdict is NO-GO")
        go = False
        ci = {"mean": np.nan, "se": np.nan, "lower": np.nan,
              "upper": np.nan, "df": 4, "tcrit": np.nan}
        dm = {"statistic": np.nan, "p_value": np.nan, "n": 5}
    else:
        ci = metrics_mod.t_confidence_interval(d, alpha=0.05)
        dm = metrics_mod.diebold_mariano(d)
        go = bool(ci["mean"] > 0.0 and ci["lower"] > 0.0
                  and 0.10 <= selection_rate <= 0.40)
    _log(f"[evaluate] primary: mean(d)={ci['mean']:+.6f} "
         f"95% CI [{ci['lower']:+.6f}, {ci['upper']:+.6f}]")
    _log(f"[evaluate] DM stat={dm['statistic']:+.4f} p={dm['p_value']:.4f}")
    _log(f"[evaluate] pooled selection rate={selection_rate:.4f} "
         f"(guard [0.10, 0.40])")
    _log(f"[evaluate] verdict: {'GO' if go else 'NO-GO'}")

    # 95% MCS (reported, not a gate) on selected-row squared errors,
    # with the actual selected rows' t0s for the day-block bootstrap.
    # (As a Series: the MCS helper calls .dt on its input.)
    mcs_out = None
    if se_frames:
        loss_df = pd.concat(se_frames, ignore_index=True)
        t0_all = pd.Series(
            pd.to_datetime(np.concatenate(t0_selected_list)))
        mcs_out = mcs_mod.mcs(loss_df, t0_all, log=_log)

    # DSR honesty metric (reported, not a gate): per-row MSE gains of
    # the challenger over the constant as pseudo-returns, freq=1,
    # pooled dev selected test rows.
    dsr_report = None
    if se_frames:
        loss_df = pd.concat(se_frames, ignore_index=True)
        dsr_report = dsr_mod.dsr_report(
            {"challenger": (loss_df["constant"].to_numpy()
                            - loss_df["challenger"].to_numpy())},
            freq=1)
        _log("[evaluate] DSR report (reported, not a gate):")
        _log(dsr_report.to_string(index=False))

    results = {
        "challenger": "lgbm_gated",
        "primary_baseline": "constant",
        "secondary_baseline": "garch_implied",
        "folds": fold_rows,
        "d": [float(v) for v in d],
        "d_garch": [float(v) for v in d_garch_list],
        "ci95": {k: (float(v) if np.isfinite(v) else None)
                 for k, v in ci.items()},
        "diebold_mariano": {k: (float(v) if np.isfinite(v) else None)
                            for k, v in dm.items()},
        "selection_rate": float(selection_rate),
        "n_selected_total": int(n_selected_total),
        "n_test_total": int(n_test_total),
        "mcs": None if mcs_out is None else {
            "survivors": mcs_out["survivors"],
            "eliminated": [(m, float(p)) for m, p in mcs_out["eliminated"]],
            "mcs_p": {m: float(p) for m, p in mcs_out["mcs_p"].items()},
        },
        "dsr": None if dsr_report is None else dsr_report.to_dict("records"),
        "verdict": "GO" if go else "NO-GO",
        "n_garch_fallbacks": n_fallbacks,
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
    """Write the working docs/CONFMAG_RESULTS.md (committed in results PR)."""
    lines: list[str] = []
    a = lines.append
    a("# CONFMAG campaign results: conformal-gated |excess_3d| vs constant (working document)")
    a("")
    a("Pre-registration: `docs/CONFMAG_PREREGISTRATION.md` (frozen, binding). "
      "This document is the working record; the results PR carries the "
      "final version.")
    a("")
    a("## Configuration")
    a("")
    a("- Challenger: **lgbm_gated** = LightGBM L2 regressor (frozen VOL "
      "hyperparameters) on the 34 frozen HAR+VIX features, scored ONLY on "
      "the CQR-selected subset")
    a("- Primary baseline: **constant** = per-fold train mean |excess_3d|, "
      "scored on the same selected subset")
    a("- Secondary baseline: **garch_implied** = sqrt(3/5) * garch_vol_5d * "
      "sqrt(2/pi)")
    a("- Conformal: CQR, LightGBM tau=0.1/0.9, alpha=0.2, temporal 20% "
      "calibration split per fold; selection = width <= 25th percentile "
      "of calibration widths")
    a("- Validation: 5-fold expanding purged walk-forward on t0, seed 7, "
      "min_train 50, 3-day horizon purge, 3-day embargo")
    a("- Primary metric: MSE on |excess_3d| over the selected subset")
    a(f"- Dev rows: {results['dev_rows']} "
      f"({results['dev_rows_after_nan_drop']} after NaN-feature drop)")
    a(f"- Selected: {results['n_selected_total']} / "
      f"{results['n_test_total']} test rows "
      f"(rate {results['selection_rate']:.4f}, guard [0.10, 0.40])")
    a(f"- GARCH naive fallbacks on test rows: {results['n_garch_fallbacks']} "
      f"/ {results['n_test_total']}")
    a("")
    a("## Per-fold MSE (|excess_3d|, selected subset)")
    a("")
    a("| fold | test range | n_train | n_test | selected | sel rate | "
      "coverage | MSE challenger | MSE constant | MSE garch_implied | "
      "d = const - challenger |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results["folds"]:
        a(f"| {r['fold']} | {r['test_start']}..{r['test_end']} | {r['n_train']} | "
          f"{r['n_test']} | {r['n_selected']} | {r['selection_rate']:.3f} | "
          f"{r['coverage']:.3f} | {r['mse_chall']:.6f} | {r['mse_const']:.6f} | "
          f"{r['mse_garch']:.6f} | {r['d']:+.6f} |")
    a("")
    a("## Verdict computation (frozen conjunctive rule)")
    a("")
    ci = results["ci95"]
    dm = results["diebold_mariano"]
    a("- Primary: d_k = MSE(constant) - MSE(challenger) per fold: "
      f"{', '.join(f'{v:+.6f}' for v in results['d'])}")
    a(f"- mean(d) = {ci['mean']:+.6f}, 95% two-sided Student-t CI: "
      f"[{ci['lower']:+.6f}, {ci['upper']:+.6f}]")
    a(f"- Diebold-Mariano: stat = {dm['statistic']:+.4f}, "
      f"p = {dm['p_value']:.4f}")
    a(f"- Pooled selection rate = {results['selection_rate']:.4f} "
      f"(guard [0.10, 0.40])")
    a(f"- GO iff mean(d) > 0 AND CI lower > 0 AND rate in guard: "
      f"**{results['verdict']}**")
    a("")
    if results["mcs"] is not None:
        a("## Model Confidence Set (95%, reported, not a gate)")
        a("")
        mcs = results["mcs"]
        a(f"- Survivors: {', '.join(mcs['survivors'])}")
        if mcs["eliminated"]:
            a("- Eliminated (in order): " +
              ", ".join(f"{m} (p={p:.4f})" for m, p in mcs["eliminated"]))
        a("")
    if results["dsr"] is not None:
        a("## Selection honesty (DSR, reported, not a gate)")
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
    from signal_lab.confmag import cqr as cqr_mod
    from signal_lab.confmag import features as confmag_feat
    from signal_lab.vol import frame as frame_mod
    from signal_lab.vol import garch as garch_mod
    from signal_lab.vol import prices as prices_mod
    from signal_lab.vol.run_campaign import _returns_by_ticker

    frame = _load_frame()
    dev = frame_mod.dev_frame(frame)
    dev_clean, _ = confmag_feat_drop_nan(dev, log=_log)
    test = frame_mod.test_frame(frame)
    test_clean, n_dropped = confmag_feat_drop_nan(test, log=_log)
    _log(f"[confirm] test rows: {len(test)} "
         f"({len(test_clean)} after NaN drop)")

    cols = confmag_feat.FEATURE_COLUMNS
    y_dev = dev_clean["target"].to_numpy(dtype=float)
    X_dev = dev_clean[cols]
    X_test = test_clean[cols]
    y_test = test_clean["target"].to_numpy(dtype=float)

    # Temporal proper-train / calibration split of full dev.
    proper_pos, cal_pos = cqr_mod.temporal_proper_cal(dev_clean["t0"])
    point_model = cqr_mod.fit_point_model(X_dev, pd.Series(y_dev))
    point_fc = cqr_mod.predict_point(point_model, X_test)
    m_lo, m_hi = cqr_mod.fit_quantile_models(
        X_dev.iloc[proper_pos], pd.Series(y_dev[proper_pos]))
    cal_out = cqr_mod.conformalize(
        m_lo, m_hi, X_dev.iloc[cal_pos], pd.Series(y_dev[cal_pos]))
    test_out = cqr_mod.predict_intervals(
        m_lo, m_hi, X_test, cal_out["q_hat"])
    selected = np.isfinite(test_out["widths"]) & (
        test_out["widths"] <= cal_out["threshold"])
    n_sel = int(selected.sum())
    _log(f"[confirm] selected {n_sel}/{len(test_clean)} "
         f"({n_sel / len(test_clean):.4f})")

    prices = prices_mod.load_or_download_prices(VOL_PRICES_PARQUET, log=_log)
    returns_by_ticker = _returns_by_ticker(prices)
    test_rows = test_clean[["ticker", "t0"]].reset_index(drop=True)
    garch_panel = garch_mod.forecast_panel(
        test_rows, returns_by_ticker, n_jobs=args.jobs,
        cache_path=GARCH_PANEL_TEST_PARQUET, log=_log)
    garch_implied = (_GARCH_3D_SCALE
                     * garch_panel["garch_vol"].to_numpy(dtype=float)
                     * _SQRT_2_OVER_PI)

    const_fc = float(np.mean(y_dev))
    ys = y_test[selected]
    mse_chall = float(np.mean((ys - point_fc[selected]) ** 2))
    mse_const = float(np.mean((ys - const_fc) ** 2))
    mse_garch = float(np.mean((ys - garch_implied[selected]) ** 2))
    _log(f"[confirm] test MSE: challenger={mse_chall:.6f} "
         f"constant={mse_const:.6f} garch_implied={mse_garch:.6f}")

    results = json.loads(RESULTS_JSON.read_text())
    section = [
        "",
        "## Test-period evaluation (single authorized run)",
        "",
        f"- Test rows scored: {len(test_clean)}; selected {n_sel} "
        f"(rate {n_sel / len(test_clean):.4f}; guard [0.10, 0.40])",
        f"- GARCH naive fallbacks on test rows: "
        f"{int(garch_panel['garch_fallback'].sum())}",
        "",
        "| arm | test MSE (selected) |",
        "| --- | --- |",
        f"| lgbm_gated | {mse_chall:.6f} |",
        f"| constant | {mse_const:.6f} |",
        f"| garch_implied | {mse_garch:.6f} |",
        "",
    ]
    with RESULTS_MD.open("a") as f:
        f.write("\n".join(section))
    results["test_eval"] = {
        "n_test_rows": len(test_clean),
        "n_selected": n_sel,
        "selection_rate": float(n_sel / len(test_clean)),
        "mse_challenger": mse_chall,
        "mse_constant": mse_const,
        "mse_garch_implied": mse_garch,
        "n_garch_fallbacks": int(garch_panel["garch_fallback"].sum()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    _log(f"[confirm] appended test evaluation to {RESULTS_MD}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="run_campaign.py",
                                description="CONFMAG campaign runner")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-frame",
                       help="build |excess_3d| frame + HAR + VIX features")
    b.add_argument("--force-download", action="store_true")
    b.set_defaults(func=cmd_build_frame)

    e = sub.add_parser("evaluate", help="5-fold walk-forward with CQR gating")
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
