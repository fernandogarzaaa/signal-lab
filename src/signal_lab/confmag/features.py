"""CONFMAG frame and target: |excess_3d| = absolute 3-day abnormal return.

Target spec (docs/CONFMAG_PREREGISTRATION.md, frozen):
- excess_3d(t0) = sum_{h=1..3} (r_i(t0+h) - r_SPY(t0+h)), daily log
  returns from split/dividend-adjusted closes; target = |excess_3d|.
- One row per (ticker, trading day), ALL ticker-days (SPY excluded
  from rows; it is the market leg for the abnormal return).
- Rows whose 3-day forward window (t0, t0+3] is partial are DROPPED,
  never filled. A labeled row whose window contains a NaN close
  raises DataQualityError (same fail-loud contract as the VOL frame).
- Features use only information knowable at t0's close. The label
  window never enters a feature.
- ``periods.check_no_confirmation`` is enforced on t0 on EVERY build.

The 34 HAR+VIX features are the frozen HARVIX definitions, added by
the caller via vol.features.add_features + harvix.features.
add_vix_features (same as the DISPVOL/ATTRVOL builds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.harvix import features as harvix_feat
from signal_lab.validation import periods
from signal_lab.vol.prices import DataQualityError
from signal_lab.vol.universe import MARKET_LEG

HORIZON_DAYS = 3

# Arm column set: the frozen 34 HAR+VIX features (the novelty under
# test is the conformal gating, not the features).
FEATURE_COLUMNS: list[str] = list(harvix_feat.HARVIX_COLUMNS)

FRAME_COLUMNS = ["ticker", "t0", "t1", "published_at", "target"]


def build_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the full (ticker, trading day) frame with |excess_3d| targets."""
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    tickers = [t for t in prices["ticker"].unique()]
    if MARKET_LEG not in tickers:
        raise ValueError(f"[confmag] market leg {MARKET_LEG} missing")
    # SPY closes keyed by date for the abnormal-return leg.
    spy = prices[prices["ticker"] == MARKET_LEG].set_index("date")["adj_close"]
    all_rows: list[dict] = []
    for ticker, grp in prices.groupby("ticker"):
        if ticker == MARKET_LEG:
            continue
        dates = grp["date"].to_numpy()
        closes = grp["adj_close"].to_numpy(dtype=float)
        n = len(dates)
        if n <= HORIZON_DAYS:
            continue
        for i in range(n - HORIZON_DAYS):
            window_closes = closes[i:i + HORIZON_DAYS + 1]
            if not np.all(np.isfinite(window_closes)):
                raise DataQualityError(
                    f"missing close inside the label window of labeled row "
                    f"({ticker}, t0={pd.Timestamp(dates[i]).date().isoformat()}); "
                    "refusing to label it"
                )
            r_i = np.log(window_closes[1:]) - np.log(window_closes[:-1])
            spy_closes = spy.reindex(
                pd.DatetimeIndex(dates[i:i + HORIZON_DAYS + 1])).to_numpy(
                    dtype=float)
            if not np.all(np.isfinite(spy_closes)):
                raise DataQualityError(
                    f"missing SPY close inside the label window of "
                    f"({ticker}, t0={pd.Timestamp(dates[i]).date().isoformat()})"
                )
            r_spy = np.log(spy_closes[1:]) - np.log(spy_closes[:-1])
            excess = float(np.sum(r_i - r_spy))
            t0 = pd.Timestamp(dates[i]).date()
            t1 = pd.Timestamp(dates[i + HORIZON_DAYS]).date()
            all_rows.append({
                "ticker": str(ticker),
                "t0": t0,
                "t1": t1,
                "published_at": pd.Timestamp(t0, tz="UTC"),
                "target": abs(excess),
            })
    frame = pd.DataFrame(all_rows, columns=FRAME_COLUMNS)
    if len(frame) == 0:
        raise DataQualityError("frame build produced zero rows")
    periods.check_no_confirmation(frame["t0"], "confmag.build_frame")
    if (frame["target"] < 0).any():
        raise ValueError("[confmag] negative |excess_3d|; bug")
    return frame
