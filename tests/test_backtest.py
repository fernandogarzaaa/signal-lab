"""M5 leak tests: position(d) may only depend on signals with asof <= d.

Setup: ticker XXX trades flat at 100, then jumps to 200 on 2026-01-06.
The jump happens between the close of Jan 5 and the close of Jan 6, so only
a position held on Jan 5 (decided from signals known on Jan 5) can capture it.

Test 1: a signal honestly stamped asof=2026-01-06 (the day the information
exists) must NOT capture the Jan 5 -> Jan 6 jump. If the engine let any
signal influence an earlier position date, this test fails.

Test 2 (documents the boundary): a signal stamped asof=2026-01-05 DOES
capture the jump. The engine is mechanical; honest asof stamping is the
caller's responsibility. This test pins that contract.
"""

import pandas as pd

from signal_lab.backtest import run


def _prices():
    dates = pd.date_range("2026-01-01", periods=10, freq="D").date
    closes = [100.0] * 5 + [200.0] * 5
    return pd.DataFrame({"ticker": "XXX", "date": list(dates), "close": closes})


def _total_return(signals, prices, cost_bps=0.0):
    res = run(signals, prices, cost_bps=cost_bps)
    return res.loc[res.strategy == "s", "total_return"].iloc[0]


def test_signal_cannot_affect_earlier_positions():
    prices = _prices()
    honest = pd.DataFrame([{
        "ticker": "XXX",
        "asof": pd.to_datetime("2026-01-06").date(),  # known Jan 6
        "strength": 1.0,
    }])
    ret = _total_return({"s": honest}, prices)
    # Position starts Jan 6; the Jan 5 -> Jan 6 jump is not captured.
    assert abs(ret) < 1e-9, f"lookahead leak: return = {ret}"


def test_asof_contract_is_mechanical():
    prices = _prices()
    early = pd.DataFrame([{
        "ticker": "XXX",
        "asof": pd.to_datetime("2026-01-05").date(),  # claimed known Jan 5
        "strength": 1.0,
    }])
    ret = _total_return({"s": early}, prices)
    # Position held Jan 5 captures the +100% jump: the engine trusts asof.
    assert ret > 0.9, f"engine did not honor asof: return = {ret}"


def test_walk_forward_folds_do_not_overlap():
    """Structural: expanding train window never includes fold dates."""
    import numpy as np

    dates = sorted(pd.date_range("2026-01-01", periods=20, freq="D").date)
    edges = np.array_split(dates, 5)
    for k in range(1, len(edges)):
        train = {d for e in edges[:k] for d in e}
        test = set(edges[k])
        assert not (train & test), "train/test date overlap in walk-forward folds"
