"""Unit tests for calibration + abstention (0.3.0 WS4). No DB, no network."""

import numpy as np
import pandas as pd

from signal_lab.models.calibration import (
    abstention_curve,
    apply_calibrators,
    expected_calibration_error,
    fit_calibrators,
    reliability_data,
    run_calibration,
)


def _frame(n=300, start="2026-01-01"):
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=n, freq="D", tz="UTC")
    texts = [
        "apple stock rises on earnings beat" if i % 3 else "market flat today"
        for i in range(n)
    ]
    labels = (rng.random(n) < 0.15).astype(int)
    return pd.DataFrame({"published_at": dates, "text": texts, "label": labels})


def test_ece_zero_for_perfectly_calibrated():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p, n_bins=2) == 0.0


def test_ece_maximal_for_perfectly_wrong():
    y = np.array([0, 0, 1, 1])
    p = np.array([1.0, 1.0, 0.0, 0.0])
    assert expected_calibration_error(y, p, n_bins=2) == 1.0


def test_reliability_bins_cover_unit_interval():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    rows = reliability_data(y, p, n_bins=5)
    assert len(rows) == 5
    assert rows[0]["bin"][0] == 0.0 and rows[-1]["bin"][1] == 1.0
    assert sum(r["count"] for r in rows) == 4


def test_calibrators_improve_ece_on_biased_proba():
    rng = np.random.default_rng(1)
    # Systematically overconfident proba: true rate 0.2, predicted ~0.8.
    y = (rng.random(400) < 0.2).astype(int)
    p = np.clip(rng.normal(0.8, 0.05, 400), 0, 1)
    before = expected_calibration_error(y, p)
    cal = fit_calibrators(y[:200], p[:200])
    after = {
        k: expected_calibration_error(y[200:], v[200:])
        for k, v in apply_calibrators(cal, p).items() if k != "raw"
    }
    assert before > 0.3
    assert min(after.values()) < before


def test_abstention_curve_monotone_coverage():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.9, 0.9, 0.1, 0.1, 0.55, 0.55])
    curve = abstention_curve(y, p, thresholds=(0.1, 0.5, 0.8))
    covs = [r["coverage"] for r in curve]
    assert covs == sorted(covs, reverse=True)
    assert curve[0]["coverage"] == 1.0


def test_abstention_acts_on_positive_proba():
    # tau gates P(y=1): only the two 0.9 articles are kept at tau=0.5.
    y = np.array([0, 1, 0, 1])
    p = np.array([0.9, 0.9, 0.1, 0.1])
    curve = abstention_curve(y, p, thresholds=(0.5,))
    assert curve[0]["n_kept"] == 2
    assert curve[0]["coverage"] == 0.5


def test_run_calibration_structure():
    df = _frame()
    result = run_calibration(df, log=lambda *a, **k: None)
    assert set(result["variants"]) == {"raw", "platt", "isotonic"}
    for name, v in result["variants"].items():
        assert 0.0 <= v["ece"] <= 1.0
        assert len(v["reliability"]) == 10
    assert result["best_calibrated"] in result["variants"]
    assert set(result["abstention"]) == {"raw", "platt", "isotonic"}
    curve = result["abstention"]["raw"]
    assert [r["tau"] for r in curve] == [0.10, 0.15, 0.20, 0.30, 0.50]
    assert result["split"]["n_train"] + result["split"]["n_calibrate"] + \
        result["split"]["n_test"] == len(df)
