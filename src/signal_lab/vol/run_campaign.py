"""VOL campaign runner: build-frame | select | evaluate | confirm-test-eval | text-attempt.

- build-frame: download/cache prices, run quality gates, build the
  (ticker, trading day) frame with frozen targets + HAR features, cache
  locally (data/vol_frame.parquet, data/vol_calendar.csv). The parquet
  cache is NOT committed.
- select: single purged train/validation split inside dev; ridge vs
  LightGBM on validation QLIKE; DSR honesty metric; writes
  data/vol_selection.json (local).
- evaluate: 5-fold expanding purged walk-forward (seed 7, min_train 50,
  5-day embargo) comparing champion vs garch11 vs naive, paired per fold
  on QLIKE; computes the frozen GO/NO-GO verdict; writes
  docs/VOL_RESULTS.md (working doc, committed in the results PR).
- confirm-test-eval: ONLY with --confirm-test-eval (and only after a dev
  GO): trains the champion on full dev, scores the untouched test period,
  appends to docs/VOL_RESULTS.md.
- text-attempt: one GDELT probe with the polite checkpointed client. On
  throttle/block: logs a deviation with evidence and SKIPS the text arms
  (no proxies, no evasion). On success: reports reachability; the
  exploratory text-arm build is then implemented in Phase 4.

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
FRAME_PARQUET = DATA_DIR / "vol_frame.parquet"
CALENDAR_CSV = DATA_DIR / "vol_calendar.csv"
SELECTION_JSON = DATA_DIR / "vol_selection.json"
RESULTS_JSON = DOCS_DIR / "vol_results.json"
RESULTS_MD = DOCS_DIR / "VOL_RESULTS.md"


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
        raise ValueError("NaN naive forecasts; refusing to score")
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
    from signal_lab.vol import features as feat_mod
    from signal_lab.vol import frame as frame_mod
    from signal_lab.vol import prices as prices_mod
    from signal_lab.vol import universe as universe_mod

    tickers = universe_mod.load_universe(UNIVERSE_CSV)
    _log(f"[build] universe: {len(tickers)} tickers")
    prices = prices_mod.load_or_download_prices(
        PRICES_PARQUET, tickers=tickers,
        force_download=args.force_download, log=_log)
    frame = frame_mod.build_frame(prices)
    _log(f"[build] frame rows: {len(frame)} "
         f"(t0 {frame['t0'].min()}..{frame['t0'].max()})")
    frame = feat_mod.add_features(frame, prices)
    _log(f"[build] features: {len(feat_mod.FEATURE_COLUMNS)} columns")
    frame.to_parquet(FRAME_PARQUET, index=False)
    cal = frame_mod.trading_calendar(prices)
    pd.DataFrame({"date": [d.date().isoformat() for d in cal]}).to_csv(
        CALENDAR_CSV, index=False)
    _log(f"[build] wrote {FRAME_PARQUET} ({len(frame)} rows) and "
         f"{CALENDAR_CSV} ({len(cal)} days)")


def cmd_select(args) -> None:
    from signal_lab.validation.splits import WalkForwardConfig, single_purged_split
    from signal_lab.stats import deflated_sharpe as dsr_mod
    from signal_lab.vol import features as feat_mod
    from signal_lab.vol import frame as frame_mod
    from signal_lab.vol import metrics as metrics_mod
    from signal_lab.vol import models as models_mod
    from signal_lab.vol import prices as prices_mod

    frame = _load_frame()
    dev = frame_mod.dev_frame(frame)
    _log(f"[select] dev rows: {len(dev)}")
    clean, n_dropped = feat_mod.drop_nan_features(dev, log=_log)
    trading_days = _load_calendar_ordinals()
    config = WalkForwardConfig(n_splits=5, window="expanding", horizon_days=5,
                               embargo_days=5, min_train=50, seed=7)
    split = single_purged_split(clean, config, trading_days, valid_fraction=0.2)
    _log(f"[select] validation {split['valid_start']}..{split['valid_end']}, "
         f"{len(split['train_idx'])} train / {len(split['valid_idx'])} valid rows, "
         f"{split['n_purged']} purged")

    X = clean[feat_mod.FEATURE_COLUMNS]
    y = clean["target"]
    Xtr, ytr = X.iloc[split["train_idx"]], y.iloc[split["train_idx"]]
    Xva, yva = X.iloc[split["valid_idx"]], y.iloc[split["valid_idx"]]

    qlikes: dict[str, float] = {}
    forecasts: dict[str, np.ndarray] = {}
    for name in models_mod.CHAMPION_CHOICES:
        fitted = models_mod.fit_arm(name, Xtr, ytr)
        fc = models_mod.predict_arm(name, fitted, Xva)
        forecasts[name] = fc
        qlikes[name] = metrics_mod.mean_qlike(yva.to_numpy() ** 2, fc ** 2)
        _log(f"[select] {name}: validation QLIKE = {qlikes[name]:.6f}")

    champion = min(qlikes, key=lambda k: qlikes[k])
    _log(f"[select] champion: {champion}")

    # DSR honesty metric (reported, not a gate): per-row QLIKE gains over
    # the naive persistence baseline as pseudo-returns, freq=1 (no
    # annualization; rows are ticker-days, not a return time series).
    prices = prices_mod.load_or_download_prices(PRICES_PARQUET, log=_log)
    returns_by_ticker = _returns_by_ticker(prices)
    valid_rows = clean.iloc[split["valid_idx"]].reset_index(drop=True)
    naive_vol = _naive_forecasts(valid_rows, returns_by_ticker)
    naive_q = metrics_mod.qlike(yva.to_numpy() ** 2, naive_vol ** 2)
    variant_returns = {
        name: (naive_q - metrics_mod.qlike(yva.to_numpy() ** 2, forecasts[name] ** 2))
        for name in models_mod.CHAMPION_CHOICES
    }
    dsr_report = dsr_mod.dsr_report(variant_returns, freq=1)
    _log("[select] DSR report (reported, not a gate):")
    _log(dsr_report.to_string(index=False))

    selection = {
        "champion": champion,
        "validation_qlike": qlikes,
        "validation_rows": int(len(split["valid_idx"])),
        "train_rows": int(len(split["train_idx"])),
        "n_purged": int(split["n_purged"]),
        "valid_start": split["valid_start"],
        "valid_end": split["valid_end"],
        "n_nan_feature_rows_dropped": int(n_dropped),
        "dev_rows": int(len(dev)),
        "ridge_hyperparams": {"alpha": models_mod.RIDGE_ALPHA},
        "lightgbm_hyperparams": models_mod.LGBM_PARAMS,
        "dsr": dsr_report.to_dict("records"),
        "dsr_note": ("per-row QLIKE gains over naive as pseudo-returns, "
                     "freq=1; reported only, not a gate"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    SELECTION_JSON.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_JSON.write_text(json.dumps(selection, indent=2, default=str))
    _log(f"[select] wrote {SELECTION_JSON}")


def _evaluate_fold(dev: pd.DataFrame, split: dict, champion: str,
                   feature_cols: list[str]):
    """Fit the champion on one fold's train, score its test rows."""
    from signal_lab.vol import metrics as metrics_mod
    from signal_lab.vol import models as models_mod

    X = dev[feature_cols]
    y = dev["target"]
    fitted = models_mod.fit_arm(
        champion, X.iloc[split["train_idx"]], y.iloc[split["train_idx"]])
    fc = models_mod.predict_arm(champion, fitted, X.iloc[split["test_idx"]])
    return fc


def cmd_evaluate(args) -> None:
    from signal_lab.validation.splits import WalkForwardConfig, make_splits
    from signal_lab.vol import features as feat_mod
    from signal_lab.vol import frame as frame_mod
    from signal_lab.vol import garch as garch_mod
    from signal_lab.vol import metrics as metrics_mod
    from signal_lab.vol import prices as prices_mod

    if not SELECTION_JSON.exists():
        raise FileNotFoundError(
            f"{SELECTION_JSON} missing; run select first")
    selection = json.loads(SELECTION_JSON.read_text())
    champion = selection["champion"]
    _log(f"[evaluate] champion from selection: {champion}")

    frame = _load_frame()
    dev = frame_mod.dev_frame(frame)
    clean, n_dropped = feat_mod.drop_nan_features(dev, log=_log)
    trading_days = _load_calendar_ordinals()
    config = WalkForwardConfig(n_splits=5, window="expanding", horizon_days=5,
                               embargo_days=5, min_train=50, seed=7)
    splits = make_splits(clean, config, trading_days)
    _log(f"[evaluate] {len(splits)} folds")

    prices = prices_mod.load_or_download_prices(PRICES_PARQUET, log=_log)
    returns_by_ticker = _returns_by_ticker(prices)

    test_rows = pd.concat(
        [clean.iloc[s["test_idx"]][["ticker", "t0"]] for s in splits],
        ignore_index=True)
    garch_panel = garch_mod.forecast_panel(
        test_rows, returns_by_ticker, n_jobs=args.jobs, log=_log)
    naive_vol = _naive_forecasts(test_rows, returns_by_ticker)
    n_fallbacks = int(garch_panel["garch_fallback"].sum())

    fold_table: list[dict] = []
    offset = 0
    for s in splits:
        n_test = len(s["test_idx"])
        sl = slice(offset, offset + n_test)
        fold_test = clean.iloc[s["test_idx"]]
        yt = fold_test["target"].to_numpy()
        champ_fc = _evaluate_fold(clean, s, champion, feat_mod.FEATURE_COLUMNS)
        garch_fc = garch_panel["garch_vol"].to_numpy()[sl]
        naive_fc = naive_vol[sl]
        q_champ = metrics_mod.mean_qlike(yt ** 2, champ_fc ** 2)
        q_garch = metrics_mod.mean_qlike(yt ** 2, garch_fc ** 2)
        q_naive = metrics_mod.mean_qlike(yt ** 2, naive_fc ** 2)
        fold_table.append({
            "fold": s["fold"],
            "test_start": s["test_start"],
            "test_end": s["test_end"],
            "n_train": int(len(s["train_idx"])),
            "n_test": int(n_test),
            "n_purged": int(s["n_purged"]),
            "n_embargoed": int(s["n_embargoed"]),
            "qlike_champion": q_champ,
            "qlike_garch11": q_garch,
            "qlike_naive": q_naive,
            "mse_var_champion": metrics_mod.mse_variance(yt ** 2, champ_fc ** 2),
            "mse_var_garch11": metrics_mod.mse_variance(yt ** 2, garch_fc ** 2),
            "mae_vol_champion": metrics_mod.mae_vol(yt, champ_fc),
            "mae_vol_garch11": metrics_mod.mae_vol(yt, garch_fc),
            "d": q_garch - q_champ,
        })
        offset += n_test
        _log(f"[evaluate] fold {s['fold']}: QLIKE champ={q_champ:.6f} "
             f"garch={q_garch:.6f} naive={q_naive:.6f} d={q_garch - q_champ:+.6f}")

    d = np.array([r["d"] for r in fold_table])
    ci = metrics_mod.t_confidence_interval(d, alpha=0.05)
    dm = metrics_mod.diebold_mariano(d)
    go = bool(ci["mean"] > 0.0 and ci["lower"] > 0.0)
    _log(f"[evaluate] mean(d)={ci['mean']:+.6f} se={ci['se']:.6f} "
         f"95% CI [{ci['lower']:+.6f}, {ci['upper']:+.6f}] df={ci['df']}")
    _log(f"[evaluate] DM stat={dm['statistic']:+.4f} p={dm['p_value']:.4f}")
    _log(f"[evaluate] verdict: {'GO' if go else 'NO-GO'}")

    results = {
        "champion": champion,
        "selection": selection,
        "folds": fold_table,
        "d": [float(v) for v in d],
        "ci95": {k: float(v) for k, v in ci.items()},
        "diebold_mariano": {k: float(v) for k, v in dm.items()},
        "verdict": "GO" if go else "NO-GO",
        "n_garch_fallbacks": n_fallbacks,
        "n_test_rows_total": int(sum(r["n_test"] for r in fold_table)),
        "dev_rows": int(len(dev)),
        "dev_rows_after_nan_drop": int(len(clean)),
        "n_nan_feature_rows_dropped": int(n_dropped),
        "deviations": [],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    _log(f"[evaluate] wrote {RESULTS_JSON}")
    _write_results_md(results)
    _log(f"[evaluate] wrote {RESULTS_MD}")


def _write_results_md(results: dict) -> None:
    """Write the working docs/VOL_RESULTS.md (committed in the results PR)."""
    lines: list[str] = []
    a = lines.append
    a("# VOL campaign results: volatility prediction (working document)")
    a("")
    a("Pre-registration: `docs/VOL_PREREGISTRATION.md` (frozen, binding). "
      "This document is the working record; the committed results PR "
      "carries the final version.")
    a("")
    a("## Configuration")
    a("")
    a(f"- Champion (from single purged selection split): **{results['champion']}**")
    a("- Baseline: GARCH(1,1) MLE, trailing 252 trading days, analytic 5-day "
      "term structure; fit failures fall back to naive persistence")
    a("- Sanity arm: naive (trailing realized_vol_5d)")
    a("- Validation: 5-fold expanding purged walk-forward on t0, seed 7, "
      "min_train 50, purge j==k, 5-trading-day embargo")
    a("- Primary metric: QLIKE on variance (lower is better)")
    a(f"- Dev rows: {results['dev_rows']} "
      f"({results['dev_rows_after_nan_drop']} after NaN-feature drop)")
    a(f"- GARCH naive fallbacks on test rows: {results['n_garch_fallbacks']} "
      f"/ {results['n_test_rows_total']}")
    a("")
    a("## Per-fold QLIKE (variance)")
    a("")
    a("| fold | test range | n_train | n_test | purged | embargoed | "
      "QLIKE champion | QLIKE garch11 | QLIKE naive | d = garch - champ |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results["folds"]:
        a(f"| {r['fold']} | {r['test_start']}..{r['test_end']} | {r['n_train']} | "
          f"{r['n_test']} | {r['n_purged']} | {r['n_embargoed']} | "
          f"{r['qlike_champion']:.6f} | {r['qlike_garch11']:.6f} | "
          f"{r['qlike_naive']:.6f} | {r['d']:+.6f} |")
    a("")
    a("## Verdict computation (frozen decision rule)")
    a("")
    ci = results["ci95"]
    dm = results["diebold_mariano"]
    a(f"- d_k = QLIKE(garch11) - QLIKE(champion) per fold: "
      f"{', '.join(f'{v:+.6f}' for v in results['d'])}")
    a(f"- mean(d) = {ci['mean']:+.6f}, se = {ci['se']:.6f}, "
      f"df = {ci['df']}, t(0.975) = {ci['tcrit']:.4f}")
    a(f"- 95% two-sided Student-t CI on mean(d): "
      f"[{ci['lower']:+.6f}, {ci['upper']:+.6f}]")
    a(f"- Diebold-Mariano on fold-level QLIKE differentials (paired-t form): "
      f"stat = {dm['statistic']:+.4f}, two-sided p = {dm['p_value']:.4f}, n = {dm['n']}")
    a(f"- GO iff mean(d) > 0 AND CI lower bound > 0: **{results['verdict']}**")
    a("")
    a("## Secondary metrics (per fold, no gate authority)")
    a("")
    a("| fold | MSE-var champion | MSE-var garch11 | MAE-vol champion | MAE-vol garch11 |")
    a("| --- | --- | --- | --- | --- |")
    for r in results["folds"]:
        a(f"| {r['fold']} | {r['mse_var_champion']:.8f} | {r['mse_var_garch11']:.8f} | "
          f"{r['mae_vol_champion']:.6f} | {r['mae_vol_garch11']:.6f} |")
    a("")
    a("## Selection honesty (DSR, reported, not a gate)")
    a("")
    a(results["selection"]["dsr_note"])
    a("")
    for row in results["selection"]["dsr"]:
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
    from signal_lab.vol import features as feat_mod
    from signal_lab.vol import frame as frame_mod
    from signal_lab.vol import garch as garch_mod
    from signal_lab.vol import metrics as metrics_mod
    from signal_lab.vol import models as models_mod
    from signal_lab.vol import prices as prices_mod

    if not SELECTION_JSON.exists():
        raise FileNotFoundError(f"{SELECTION_JSON} missing; run select first")
    selection = json.loads(SELECTION_JSON.read_text())
    champion = selection["champion"]

    frame = _load_frame()
    dev = frame_mod.dev_frame(frame)
    dev_clean, _ = feat_mod.drop_nan_features(dev, log=_log)
    test = frame_mod.test_frame(frame)
    test_clean, n_dropped = feat_mod.drop_nan_features(test, log=_log)
    _log(f"[confirm] test rows: {len(test)} ({len(test_clean)} after NaN drop)")

    prices = prices_mod.load_or_download_prices(PRICES_PARQUET, log=_log)
    returns_by_ticker = _returns_by_ticker(prices)

    X = dev_clean[feat_mod.FEATURE_COLUMNS]
    fitted = models_mod.fit_arm(champion, X, dev_clean["target"])
    Xt = test_clean[feat_mod.FEATURE_COLUMNS]
    champ_fc = models_mod.predict_arm(champion, fitted, Xt)

    test_rows = test_clean[["ticker", "t0"]].reset_index(drop=True)
    garch_panel = garch_mod.forecast_panel(
        test_rows, returns_by_ticker, n_jobs=args.jobs, log=_log)
    naive_fc = _naive_forecasts(test_rows, returns_by_ticker)

    yt = test_clean["target"].to_numpy()
    q_champ = metrics_mod.mean_qlike(yt ** 2, champ_fc ** 2)
    q_garch = metrics_mod.mean_qlike(
        yt ** 2, garch_panel["garch_vol"].to_numpy() ** 2)
    q_naive = metrics_mod.mean_qlike(yt ** 2, naive_fc ** 2)
    _log(f"[confirm] test QLIKE: champ={q_champ:.6f} garch={q_garch:.6f} "
         f"naive={q_naive:.6f}")

    results = json.loads(RESULTS_JSON.read_text())
    dev_means = {
        arm: float(np.mean([r[f"qlike_{arm}"] for r in results["folds"]]))
        for arm in ("champion", "garch11", "naive")
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
        f"| champion ({champion}) | {dev_means['champion']:.6f} | {q_champ:.6f} |",
        f"| garch11 | {dev_means['garch11']:.6f} | {q_garch:.6f} |",
        f"| naive | {dev_means['naive']:.6f} | {q_naive:.6f} |",
        "",
    ]
    with RESULTS_MD.open("a") as f:
        f.write("\n".join(section))
    results["test_eval"] = {
        "n_test_rows": int(len(test_clean)),
        "t0_min": str(test_clean["t0"].min()),
        "t0_max": str(test_clean["t0"].max()),
        "qlike_champion": q_champ,
        "qlike_garch11": q_garch,
        "qlike_naive": q_naive,
        "dev_mean_qlike": dev_means,
        "n_garch_fallbacks": int(garch_panel["garch_fallback"].sum()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    _log(f"[confirm] appended test evaluation to {RESULTS_MD}")


def cmd_text_attempt(args) -> None:
    """One GDELT probe with the polite checkpointed client.

    On throttle/block: log a deviation with evidence (HTTP status,
    timestamps) and SKIP the text arms. No proxies, no evasion, no
    retries beyond the client's own backoff posture.
    """
    from gdelt_client.errors import RateLimitError

    from signal_lab.ingest import gdelt_client

    probe_query = '("Apple Inc. stock" OR "AAPL stock") sourcelang:english'
    probe_start, probe_end = "2026-06-01", "2026-06-01"
    t0 = datetime.now(timezone.utc).isoformat()
    _log(f"[text] probe {t0}: one DOC article_search for {probe_start}")
    try:
        metas = gdelt_client.fetch_news_via_client(
            probe_query, probe_start, probe_end,
            max_records=10, fetch_bodies=False, retries=1)
    except RateLimitError as exc:
        t1 = datetime.now(timezone.utc).isoformat()
        evidence = (f"GDELT DOC API probe {probe_start} hit RateLimitError "
                    f"(HTTP 429 via gdelt-client backoff) at {t1}; "
                    f"probe started {t0}; detail: {exc}")
        _log(f"[text] THROTTLED: {evidence}")
        _record_deviation(
            "TEXT ARMS SKIPPED: " + evidence + " No proxies or rate-limit "
            "evasion attempted. Text arms (TF-IDF + Loughran-McDonald) are "
            "not run in this campaign.")
        _log("[text] text arms skipped; deviation logged in VOL_RESULTS.md")
        return
    except Exception as exc:  # blocked in some other way: same policy
        t1 = datetime.now(timezone.utc).isoformat()
        evidence = (f"GDELT DOC API probe {probe_start} failed at {t1} "
                    f"(probe started {t0}): {type(exc).__name__}: {exc}")
        _log(f"[text] BLOCKED: {evidence}")
        _record_deviation("TEXT ARMS SKIPPED: " + evidence + " No proxies "
                          "or rate-limit evasion attempted.")
        _log("[text] text arms skipped; deviation logged in VOL_RESULTS.md")
        return
    _log(f"[text] probe returned {len(metas)} articles; GDELT reachable")
    _log("[text] UNEXPECTED: DOC API is reachable; the exploratory "
         "price+text arm build (TF-IDF + Loughran-McDonald, vectorizer fit "
         "on train per fold, paired tests, no gate authority) must be "
         "implemented before proceeding")
    raise SystemExit(2)


def _record_deviation(text: str) -> None:
    """Append a deviation entry to the working VOL_RESULTS.md."""
    if RESULTS_JSON.exists():
        results = json.loads(RESULTS_JSON.read_text())
        results.setdefault("deviations", []).append(text)
        RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    if RESULTS_MD.exists():
        md = RESULTS_MD.read_text()
        marker = "## Deviations\n\n"
        if marker in md:
            md = md.replace(marker, marker + f"- {text}\n", 1)
            # drop the placeholder "None." line if present
            md = md.replace(marker + f"- {text}\nNone.\n",
                            marker + f"- {text}\n", 1)
            RESULTS_MD.write_text(md)
        else:
            with RESULTS_MD.open("a") as f:
                f.write(f"\n## Deviations\n\n- {text}\n")
    else:
        _log("[text] WARNING: VOL_RESULTS.md does not exist yet; deviation "
             "kept in vol_results.json only")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="run_campaign.py",
                                description="VOL volatility campaign runner")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-frame", help="download prices, build frame+features")
    b.add_argument("--force-download", action="store_true")
    b.set_defaults(func=cmd_build_frame)

    s = sub.add_parser("select", help="single purged split: ridge vs LightGBM")
    s.set_defaults(func=cmd_select)

    e = sub.add_parser("evaluate", help="5-fold walk-forward vs garch11")
    e.add_argument("--jobs", type=int, default=1)
    e.set_defaults(func=cmd_evaluate)

    c = sub.add_parser("confirm-test-eval",
                       help="single authorized test-period evaluation")
    c.add_argument("--confirm-test-eval", action="store_true",
                   help="explicit confirmation flag (required)")
    c.add_argument("--jobs", type=int, default=1)
    c.set_defaults(func=cmd_confirm_test_eval)

    t = sub.add_parser("text-attempt", help="one GDELT probe; skip text arms if throttled")
    t.set_defaults(func=cmd_text_attempt)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
