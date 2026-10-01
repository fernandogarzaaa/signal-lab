"""Tests for cross-sectional CAR significance tests (car_tests).

Uses synthetic AR panels with known properties: a mean shift the tests
must detect, a null panel they must not flag, event-induced variance
(BMP's raison d'etre), correlated estimation windows (adj-BMP), and
the event_ar_panel builder plus the event_study wiring.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from signal_lab.stats import car_tests, event_study


def _panel(n_events: int, L: int = 3, est_days: int = 120, shift: float = 0.0,
           event_var_mult: float = 1.0, corr: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Synthetic AR panel with known properties.

    shift: mean daily AR added in the event window. event_var_mult:
    variance multiplier applied to event-window ARs only (event-induced
    variance). corr: common-factor loading in the estimation window
    (pairwise estimation correlation ~= corr**2).
    """
    rng = np.random.default_rng(seed)
    # Common factor on the same scale as the idiosyncratic noise, so the
    # pairwise estimation correlation is ~= corr**2.
    common = rng.normal(scale=0.01, size=est_days)
    rows = []
    for eid in range(n_events):
        idio = rng.normal(scale=0.01, size=est_days)
        est = np.sqrt(max(1 - corr ** 2, 0)) * idio + corr * common
        evt = shift + rng.normal(scale=0.01 * np.sqrt(event_var_mult), size=L)
        for k, a in enumerate(est):
            rows.append({"event_id": eid, "ticker": "T",
                         "event_date": date(2025, 6, 1),
                         "offset": -(est_days - k) - 2, "ar": float(a),
                         "in_event": False})
        for k, a in enumerate(evt):
            rows.append({"event_id": eid, "ticker": "T",
                         "event_date": date(2025, 6, 1),
                         "offset": k - 1, "ar": float(a), "in_event": True})
    return pd.DataFrame(rows)


def test_all_tests_reject_real_shift():
    panel = _panel(60, shift=0.015, seed=1)
    res = car_tests.cross_sectional_tests(panel, pre=1, post=1)
    assert res["n_events"] == 60
    for name in ("t", "bmp", "adj_bmp", "grank"):
        assert res[name]["p_value"] < 0.05, f"{name} failed to reject a real shift"


def test_all_tests_quiet_on_null():
    panel = _panel(200, shift=0.0, seed=2)
    res = car_tests.cross_sectional_tests(panel, pre=1, post=1)
    for name in ("t", "bmp", "adj_bmp", "grank"):
        assert res[name]["p_value"] > 0.05, f"{name} false positive on null panel"


def test_bmp_matches_hand_computation():
    scars = np.array([2.0, 2.5, 1.5, 3.0, 2.2])
    t, p = car_tests.bmp_test(scars)
    expected_t = scars.mean() / (scars.std(ddof=1) / np.sqrt(len(scars)))
    assert t == expected_t
    assert 0.0 < p < 0.05


def test_bmp_robust_to_event_induced_variance():
    # 4x event-window variance with a real mean shift: BMP standardizes
    # by ESTIMATION variance, so it must still reject.
    panel = _panel(80, shift=0.02, event_var_mult=4.0, seed=3)
    res = car_tests.cross_sectional_tests(panel, pre=1, post=1)
    assert res["bmp"]["p_value"] < 0.05
    assert res["grank"]["p_value"] < 0.05


def test_adj_bmp_shrinks_statistic_under_correlation():
    panel = _panel(60, shift=0.015, corr=0.7, seed=4)
    res = car_tests.cross_sectional_tests(panel, pre=1, post=1)
    r_bar = res["r_bar"]
    assert 0.2 < r_bar < 0.8, f"expected sizable r_bar, got {r_bar}"
    assert abs(res["adj_bmp"]["stat"]) < abs(res["bmp"]["stat"]), \
        "adj-BMP must shrink |t| when estimation windows are correlated"


def test_adj_bmp_reduces_to_bmp_at_zero_correlation():
    # Unit property: r_bar = 0 must reproduce BMP exactly.
    rng = np.random.default_rng(0)
    scars = rng.normal(loc=2.0, scale=1.0, size=40)
    t_bmp, p_bmp = car_tests.bmp_test(scars)
    t_adj, p_adj, r_used = car_tests.adj_bmp_test(scars, 0.0)
    assert t_adj == t_bmp and p_adj == p_bmp and r_used == 0.0


def test_adj_bmp_panel_without_correlation_has_small_rbar():
    panel = _panel(60, shift=0.015, seed=5)
    res = car_tests.cross_sectional_tests(panel, pre=1, post=1)
    assert abs(res["r_bar"]) < 0.15


def test_too_few_events_returns_nan_not_garbage():
    panel = _panel(2, shift=0.05, seed=6)
    res = car_tests.cross_sectional_tests(panel, pre=1, post=1)
    assert res["n_events"] == 2
    for name in ("t", "bmp", "adj_bmp", "grank"):
        assert np.isnan(res[name]["stat"]), f"{name} should be NaN with n=2"


def test_zero_estimation_variance_events_dropped():
    panel = _panel(10, shift=0.01, seed=7)
    # Freeze one event's estimation ARs: zero variance -> must be dropped.
    m = (panel["event_id"] == 0) & (~panel["in_event"])
    panel.loc[m, "ar"] = 0.001
    res = car_tests.cross_sectional_tests(panel, pre=1, post=1)
    assert res["n_events"] == 9
    assert res["n_dropped"] == 1


def _synthetic_prices(n_days: int = 400, seed: int = 9):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    frames = []
    for t in ("AAA", "BBB", "SPY"):
        rets = rng.normal(scale=0.01, size=n_days)
        close = 100 * np.exp(np.cumsum(rets))
        frames.append(pd.DataFrame({"ticker": t, "date": dates.date, "close": close}))
    prices = pd.concat(frames, ignore_index=True)
    # Events on days with full estimation history behind them.
    ev_dates = [dates[150].date(), dates[200].date(), dates[250].date(),
                dates[300].date(), dates[350].date()]
    events = pd.DataFrame(
        {"ticker": ["AAA", "BBB", "AAA", "BBB", "AAA"],
         "event_date": ev_dates, "sentiment": [0.9] * 5}
    )
    return events, prices


def test_event_ar_panel_builder():
    events, prices = _synthetic_prices()
    panel = car_tests.event_ar_panel(events, prices, pre=1, post=1, est_days=120)
    assert set(panel.columns) == {"event_id", "ticker", "event_date", "offset",
                                  "ar", "in_event"}
    # 120 estimation + 3 event-window days per event.
    counts = panel.groupby("event_id").size()
    assert (counts == 123).all()
    assert panel.groupby("event_id")["in_event"].sum().eq(3).all()


def test_event_ar_panel_drops_events_without_history():
    events, prices = _synthetic_prices()
    early = pd.DataFrame({"ticker": ["AAA"], "event_date": [pd.bdate_range("2024-01-01", periods=400)[5].date()],
                          "sentiment": [0.9]})
    panel = car_tests.event_ar_panel(pd.concat([events, early], ignore_index=True),
                                     prices, pre=1, post=1, est_days=120)
    assert panel["event_id"].nunique() == 5  # the early event is dropped


def test_event_study_wires_new_columns():
    events, prices = _synthetic_prices()
    res = event_study(events, prices)
    for col in ("bmp_stat", "bmp_p", "adj_bmp_stat", "adj_bmp_p",
                "adj_bmp_rbar", "grank_stat", "grank_p", "n_car_events"):
        assert col in res.columns, f"missing column {col}"
    # Existing columns untouched.
    for col in ("n", "mean_car", "t_stat", "p_value", "overlap_rate"):
        assert col in res.columns
