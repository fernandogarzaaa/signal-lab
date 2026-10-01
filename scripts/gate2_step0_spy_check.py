"""STEP 0a integrity check: did the SPY price gap affect gate 1?

Background: SPY prices in prices_daily covered only 2024-10-15..2026-09-30
when gate 1 ran; a later backfill added 2023-07-01..2024-10-14 (324 rows).
The label code falls back to the universe equal-weight mean return on dates
where the benchmark has no price (never NaN, never silently dropped).

This script rebuilds the ORIGINAL gate-1 frame (news fetched before the
2026-10-01 widening) twice:
  - frame_old: prices with SPY truncated to >= 2024-10-15 (pre-backfill state)
  - frame_new: prices with the full backfilled SPY (current state)
then compares every SPY-dependent column and re-runs the gate benchmark on
frame_new, comparing the primary verdict numbers to docs/GATE.md.

Usage: .venv/bin/python scripts/gate2_step0_spy_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from signal_lab.ingest import DB_PATH  # noqa: E402
from signal_lab.models.build_and_train import load_trading_calendar_days  # noqa: E402
from signal_lab.models.context_features import (  # noqa: E402
    CONTEXT_FEATURE_NAMES,
    PRICE_FEATURE_NAMES,
    build_context_features,
    build_price_features,
)
from signal_lab.models.labels import LabelConfig, apply_legacy_labeling, build_labels  # noqa: E402
from signal_lab.nlp import extract_baseline  # noqa: E402
from signal_lab.validation.targets import (  # noqa: E402
    TargetConfig,
    build_targets,
    target_column_names,
)
from signal_lab.benchmark.phase3 import run_benchmark  # noqa: E402

ORIG_NEWS_CUTOFF = "2026-10-01"  # fetched_at before the widening run
SPY_PRE_BACKFILL_START = "2024-10-15"


def build_frame(prices: pd.DataFrame, log=print) -> pd.DataFrame:
    """build_dataset pipeline, restricted to pre-widening news rows."""
    label_cfg = LabelConfig().validated()
    con = duckdb.connect(DB_PATH, read_only=True)
    news = con.execute(
        "SELECT url, title, body_snippet, published_at FROM news_raw "
        f"WHERE title <> '' AND DATE(fetched_at) < '{ORIG_NEWS_CUTOFF}'"
    ).fetchdf()
    con.close()
    log(f"[step0] original news rows: {len(news)}")

    news["pub_date"] = pd.to_datetime(news["published_at"], utc=True).dt.date
    tickers, dropped = [], 0
    news["text"] = (
        news["title"].fillna("") + " " + news["body_snippet"].fillna("")
    ).str.strip()
    for text in news["text"]:
        hits = extract_baseline(text)
        if hits:
            tickers.append(max(hits, key=lambda h: h[2])[1])
        else:
            tickers.append(None)
            dropped += 1
    news["ticker"] = tickers
    news = news.dropna(subset=["ticker"]).reset_index(drop=True)
    log(f"[step0] articles with a ticker: {len(news)} (dropped {dropped})")

    ctx = build_context_features(news, prices, benchmark=label_cfg.benchmark, log=log)
    ctx_ok = ~ctx[CONTEXT_FEATURE_NAMES].isna().any(axis=1)
    news = news[ctx_ok].reset_index(drop=True)
    ctx = ctx[ctx_ok].reset_index(drop=True)

    labeled, label_info = build_labels(news, prices, label_cfg)
    log(f"[step0] labeled rows: {label_info['n_rows']}")
    labeled["text"] = news.set_index("url").loc[labeled["url"], "text"].values
    labeled = labeled.merge(
        ctx[["url", "ctx_asof"] + CONTEXT_FEATURE_NAMES], on="url", how="left"
    )
    legacy_labeled, _ = apply_legacy_labeling(labeled, label_cfg)
    tgt_cfg = TargetConfig().validated()
    targets = build_targets(legacy_labeled[["ticker", "t0"]], prices, tgt_cfg, log=log)
    legacy_labeled = pd.concat([legacy_labeled, targets], axis=1)
    px = build_price_features(legacy_labeled, prices, benchmark=label_cfg.benchmark, log=log)
    legacy_labeled = pd.concat([legacy_labeled, px[PRICE_FEATURE_NAMES]], axis=1)
    cols = (
        ["url", "text", "ticker", "published_at", "pub_date", "t0", "t1",
         "fwd_days", "n_articles", "ticker_fwd_ret", "mkt_fwd_ret", "abn_ret",
         "label", "ctx_asof"]
        + CONTEXT_FEATURE_NAMES
        + target_column_names(tgt_cfg)
        + PRICE_FEATURE_NAMES
    )
    return legacy_labeled[cols].reset_index(drop=True)


def main() -> None:
    log = print
    con = duckdb.connect(DB_PATH, read_only=True)
    prices = con.execute(
        "SELECT ticker, date, open, high, low, close, volume FROM prices_daily"
    ).fetchdf()
    con.close()
    prices["date"] = pd.to_datetime(prices["date"]).dt.date

    prices_old = prices[
        ~((prices["ticker"] == "SPY") & (prices["date"] < pd.to_datetime(SPY_PRE_BACKFILL_START).date()))
    ].copy()
    n_spy_removed = len(prices) - len(prices_old)
    log(f"[step0] simulated pre-backfill state: removed {n_spy_removed} SPY rows")

    frame_old = build_frame(prices_old, log=lambda *a, **k: None)
    frame_new = build_frame(prices, log=lambda *a, **k: None)
    log(f"[step0] frame rows: old={len(frame_old)} new={len(frame_new)}")
    assert len(frame_old) == len(frame_new), "row count changed with SPY backfill"
    assert (frame_old["url"].values == frame_new["url"].values).all(), "row order changed"

    spy_cols = ["ticker_fwd_ret", "mkt_fwd_ret", "abn_ret", "excess_3d"] + [
        c for c in CONTEXT_FEATURE_NAMES + PRICE_FEATURE_NAMES if "spy" in c.lower()
    ]
    log(f"[step0] SPY-dependent columns checked: {spy_cols}")
    max_diff = 0.0
    n_differ = 0
    for c in spy_cols:
        if c not in frame_old.columns or c not in frame_new.columns:
            log(f"[step0] column {c} missing in one frame; skipping")
            continue
        a = frame_old[c].to_numpy(dtype=float)
        b = frame_new[c].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        differ = (~both_nan) & (np.isnan(a) | np.isnan(b) | (np.abs(a - b) > 1e-12))
        d = int(differ.sum())
        md = float(np.abs(a[~both_nan] - b[~both_nan]).max()) if (~both_nan).any() else 0.0
        n_differ += d
        max_diff = max(max_diff, md)
        if d:
            log(f"[step0] DIFFER: {c}: {d} rows differ, max abs diff {md:.6g}")
    log(f"[step0] total differing cells: {n_differ}; max abs diff: {max_diff:.6g}")

    # Fallback audit: did the old frame ever use the equal-weight fallback?
    # Recompute the market leg directly.
    from signal_lab.models.labels import _daily_returns, _market_daily_returns

    rets_old = _daily_returns(prices_old)
    mkt_old = _market_daily_returns(rets_old, "SPY")
    spy_rets = rets_old[rets_old.ticker == "SPY"].drop_duplicates("date").set_index("date")["ret"]
    fallback_dates = mkt_old.index[~mkt_old.index.isin(spy_rets.dropna().index)]
    log(f"[step0] dates where old frame used fallback market leg: {len(fallback_dates)}")
    if len(fallback_dates):
        log(f"[step0] fallback date range: {fallback_dates.min()}..{fallback_dates.max()}")

    # Re-run the gate benchmark on the corrected (current) frame.
    trading_days = load_trading_calendar_days(log=lambda *a, **k: None)
    results = run_benchmark(frame_new, trading_days, log=lambda *a, **k: None)
    gate = results["gate"]
    log("[step0] corrected-frame gate verdict: " + gate["verdict"])
    md = gate["primary_mean_diff"]
    ci = gate["primary_ci95"]
    pv = gate["primary_t_pvalue"]
    log(f"[step0] C-A mean diff: {md:.4f}")
    log(f"[step0] 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    log(f"[step0] paired t p: {pv:.4f}")

    report = {
        "n_rows": len(frame_new),
        "n_differ_cells": n_differ,
        "max_abs_diff": max_diff,
        "fallback_dates_used": len(fallback_dates),
        "verdict": gate["verdict"],
        "mean_diff": md,
        "ci": [ci[0], ci[1]],
        "p_value": pv,
    }
    out = REPO_ROOT / "data" / "artifacts" / "step0_spy_check.json"
    out.write_text(json.dumps(report, indent=2))
    log(f"[step0] report -> {out}")


if __name__ == "__main__":
    main()
