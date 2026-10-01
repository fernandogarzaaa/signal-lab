"""Gate-2 STEP 5: run the pre-registered benchmark on the widened dev data.

Frozen spec (docs/GATE_2_PREREGISTRATION.md):
- same 5 models, same 21 price features, same weak labels, same 5-fold
  expanding walk-forward with purge + 5-day embargo, seed 7
- primary model: logreg_balanced
- primary comparison: Fc-A (FinBERT text+price vs price-only), paired
  per-fold PR-AUC, Student-t 95% CI
- GO iff mean paired (Fc-A) PR-AUC difference > 0 AND 95% CI lower > 0
- confirmation period (t0 >= 2026-09-01) is NEVER touched here: the
  benchmark runs on the development period only (t0 <= 2026-06-30).

Prerequisites (run in order before this script):
1. python -m signal_lab.ingest.widen            (STEP 3 ingest, background)
2. python scripts/fill_finbert_cache.py          (STEP 4 cache for new texts)
3. python -m signal_lab.ingest.widen --check-only (quality_check passes)

Writes data/artifacts/gate2_report.json and prints the verdict.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from signal_lab.benchmark.phase3 import (  # noqa: E402
    PRIMARY_COMPARISON_GATE2,
    PRIMARY_MODEL,
    BenchmarkConfig,
    run_benchmark,
)
from signal_lab.models.build_and_train import (  # noqa: E402
    build_dataset,
    load_trading_calendar_days,
)


def main() -> None:
    art = REPO_ROOT / "data" / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    print("[gate2] building widened dev dataset ...", flush=True)
    df, _ = build_dataset(log=print)
    trading_days = load_trading_calendar_days(log=print)
    print(f"[gate2] dev rows: {len(df)}", flush=True)

    cfg = BenchmarkConfig(
        primary_comparison=PRIMARY_COMPARISON_GATE2,
    ).validated()
    print(f"[gate2] primary comparison: {cfg.primary_comparison} "
          f"(primary model: {PRIMARY_MODEL})", flush=True)

    res = run_benchmark(df, trading_days, config=cfg, log=print)

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "primary_comparison": cfg.primary_comparison,
        "primary_model": PRIMARY_MODEL,
        "n_dev_rows": len(df),
        "n_folds_completed": res["n_folds_completed"],
        "aggregate": res["aggregate"],
        "paired": res["paired"],
        "gate": res["gate"],
    }
    out_path = art / "gate2_report.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[gate2] report written to {out_path}", flush=True)

    g = res["gate"]
    print("=" * 60)
    print(f"GATE-2 VERDICT ({g['primary_comparison']}, "
          f"{g['primary_model']}): {g['verdict']}")
    print(f"  mean paired diff : {g['primary_mean_diff']:.4f}")
    lo, hi = g["primary_ci95"]
    print(f"  95% CI           : [{lo:.4f}, {hi:.4f}]")
    print(f"  t p-value        : {g['primary_t_pvalue']:.4f}")
    print("=" * 60)
    if g["verdict"] == "GO":
        print("[gate2] GO: STOP per pre-registration. Report results; "
              "test period may be evaluated at most once, only with approval.")
    else:
        print("[gate2] NO-GO: STOP per pre-registration. Write "
              "docs/NEGATIVE_RESULT.md covering BOTH gate runs honestly; "
              "no rescue experiments.")


if __name__ == "__main__":
    main()
