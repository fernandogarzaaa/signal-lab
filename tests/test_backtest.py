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


def test_expand_events_empty_keeps_numeric_dtypes():
    """Regression (0.2.0): a fold with zero events must return an empty frame
    with float64 strength. Concatenating an empty object-dtype frame upcasts a
    sibling frame's strength to object, which crashed scipy's ttest_rel."""
    from signal_lab.backtest.run_backtest import expand_events

    events = pd.DataFrame(columns=["ticker", "event_date"])
    sig = expand_events(events, [pd.to_datetime("2026-01-01").date(),
                                 pd.to_datetime("2026-01-02").date()],
                        hold_days=5)
    assert sig.empty
    assert str(sig["strength"].dtype) == "float64", sig.dtypes


def test_run_coerces_object_port_ret_before_ttest():
    """Regression (0.2.0): run() must survive an object-dtype strength column
    (as produced by concatenating an empty fold frame) instead of crashing
    inside scipy's ttest_rel."""
    prices = _prices()
    real = pd.DataFrame([{
        "ticker": "XXX",
        "asof": pd.to_datetime("2026-01-05").date(),
        "strength": 1.0,
    }])
    empty_object = pd.DataFrame({
        "ticker": pd.Series(dtype=object),
        "asof": pd.Series(dtype=object),
        "strength": pd.Series(dtype=object),
    })
    mixed = pd.concat([real, empty_object], ignore_index=True)
    # pandas>=2.2 may infer float64 on concat with an empty object frame;
    # force the hostile input explicitly so the precondition holds on any
    # pandas version (drive-by, 2026-10-02: tabpfn==2.0.6 pins pandas<3,
    # which resolves CI to pandas 2.3.3 where the bare concat infers
    # float64 and this precondition failed).
    mixed["strength"] = mixed["strength"].astype(object)
    assert mixed["strength"].dtype == object  # precondition: the hostile input
    bnh = pd.DataFrame([{
        "ticker": "XXX",
        "asof": pd.to_datetime("2026-01-01").date(),
        "strength": 1.0,
    }])
    res = run({"s": mixed, "buy_and_hold": bnh}, prices, cost_bps=0.0)
    assert not res.empty
    assert "paired_p" in res.columns
