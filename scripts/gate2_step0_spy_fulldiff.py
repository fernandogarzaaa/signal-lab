"""STEP 0a (part 2): compare EVERY column between pre/post SPY-backfill frames.

Reuses the pipeline from gate2_step0_spy_check via import.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

spec = importlib.util.spec_from_file_location(
    "spy_check", REPO_ROOT / "scripts" / "gate2_step0_spy_check.py"
)
spy_check = importlib.util.module_from_spec(spec)
sys.argv = ["gate2_step0_spy_check.py"]  # guard against argparse side effects
spec.loader.exec_module(spy_check)

from signal_lab.ingest import DB_PATH  # noqa: E402


def main() -> None:
    con = duckdb.connect(DB_PATH, read_only=True)
    prices = con.execute(
        "SELECT ticker, date, open, high, low, close, volume FROM prices_daily"
    ).fetchdf()
    con.close()
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices_old = prices[
        ~((prices["ticker"] == "SPY")
          & (prices["date"] < pd.to_datetime("2024-10-15").date()))
    ].copy()

    frame_old = spy_check.build_frame(prices_old, log=lambda *a, **k: None)
    frame_new = spy_check.build_frame(prices, log=lambda *a, **k: None)
    assert len(frame_old) == len(frame_new)
    assert (frame_old["url"].values == frame_new["url"].values).all()

    total_differ = 0
    for c in frame_old.columns:
        a = frame_old[c]
        b = frame_new[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av = a.to_numpy(dtype=float)
            bv = b.to_numpy(dtype=float)
            both_nan = np.isnan(av) & np.isnan(bv)
            differ = (~both_nan) & (
                np.isnan(av) | np.isnan(bv) | (np.abs(av - bv) > 1e-12)
            )
            d = int(differ.sum())
        else:
            d = int((a.fillna("<NA>").astype(str) != b.fillna("<NA>").astype(str)).sum())
        if d:
            print(f"DIFFER column {c}: {d} rows")
            total_differ += d
    print(f"total differing cells across ALL columns: {total_differ}")
    print(f"frame shape: {frame_new.shape}")


if __name__ == "__main__":
    main()
