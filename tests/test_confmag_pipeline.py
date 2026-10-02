"""End-to-end CONFMAG pipeline test on synthetic data (no network).

Builds a small price panel, the |excess_3d| frame, HAR + synthetic VIX
features, runs a small purged walk-forward with the frozen CQR gating
and LightGBM arms, scores MSE on the selected subset vs the constant
baseline, computes the frozen conjunctive verdict form, and verifies
the confirmation period is never touchable.
"""

import numpy as np
import pandas as pd
import pytest

from signal_lab.confmag import cqr as cqr_mod
from signal_lab.confmag import features as feat_mod
from signal_lab.confmag import run_campaign as rc
from signal_lab.harvix import features as harvix_feat
from signal_lab.validation import periods
from signal_lab.validation.periods import ConfirmationLeakError
from signal_lab.validation.splits import WalkForwardConfig, make_splits
from signal_lab.vol import features as vol_feat
from signal_lab.vol import frame as frame_mod
from signal_lab.vol import metrics as metrics_mod


def _panel(n_tickers=20, n_days=250, start="2022-01-03", seed=21):
    tickers = [f"T{i:02d}" for i in range(n_tickers)] + ["SPY"]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for t in tickers:
        rets = rng.normal(0.0003, 0.012, n_days)
        close = 100.0 * np.exp(np.cumsum(rets))
        for i, d in enumerate(dates):
            rows.append({"ticker": t, "date": d, "open": close[i],
                         "high": close[i] * 1.001, "low": close[i] * 0.999,
                         "close": close[i], "adj_close": close[i],
                         "volume": 1_000_000.0})
    return pd.DataFrame(rows)


def _vix(n_days=250, start="2022-01-03", seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    vix = np.clip(20.0 + np.cumsum(rng.normal(0, 0.5, n_days)), 5.0, None)
    vix3m = vix * (1.0 + rng.normal(0.05, 0.01, n_days))
    return pd.DataFrame({"date": dates, "vix": vix, "vix3m": vix3m})


def _built_frame():
    prices = _panel()
    frame = feat_mod.build_frame(prices)
    frame = vol_feat.add_features(frame, prices)
    frame = harvix_feat.add_vix_features(frame, _vix())
    return frame, prices


def test_frame_and_nan_drop():
    frame, _ = _built_frame()
    assert set(feat_mod.FEATURE_COLUMNS) <= set(frame.columns)
    assert (frame["target"] >= 0).all()
    clean, n_dropped = rc.confmag_feat_drop_nan(frame, log=lambda *a: None)
    assert len(clean) > 0
    assert clean[feat_mod.FEATURE_COLUMNS].notna().all().all()


def test_walk_forward_cqr_gating_and_verdict():
    frame, prices = _built_frame()
    clean, _ = rc.confmag_feat_drop_nan(frame, log=lambda *a: None)
    dev = frame_mod.dev_frame(clean)
    trading_days = frame_mod.union_trading_ordinals(prices)
    config = WalkForwardConfig(n_splits=2, window="expanding",
                               horizon_days=3, embargo_days=3,
                               min_train=20, seed=7)
    splits = make_splits(dev, config, trading_days)
    assert len(splits) == 2

    n_sel_total, n_test_total = 0, 0
    fold_d = []
    for s in splits:
        train_idx, test_idx = s["train_idx"], s["test_idx"]
        n_test = len(test_idx)
        # Fake GARCH vol (not the focus of this smoke test).
        garch_vol = np.full(n_test, 0.01)
        fr = rc._evaluate_fold(clean, train_idx, test_idx, garch_vol,
                               log=lambda *a: None)
        n_sel_total += fr["n_selected"]
        n_test_total += fr["n_test"]
        assert 0.0 <= fr["coverage"] <= 1.0
        assert fr["n_cal"] > 0
        if fr["n_selected"] > 0:
            assert np.isfinite(fr["mse_chall"])
            fold_d.append(fr["d"])

    selection_rate = n_sel_total / n_test_total
    assert 0.0 < selection_rate < 1.0
    # Frozen conjunctive verdict form.
    d = np.array(fold_d)
    if len(d) == 2 and np.all(np.isfinite(d)):
        ci = metrics_mod.t_confidence_interval(d, alpha=0.05)
        dm = metrics_mod.diebold_mariano(d)
        assert set(ci) >= {"mean", "se", "lower", "upper", "df"}
        assert set(dm) >= {"statistic", "p_value", "n"}
        verdict = bool(ci["mean"] > 0.0 and ci["lower"] > 0.0
                         and 0.10 <= selection_rate <= 0.40)
        assert isinstance(verdict, bool)


def test_confirmation_never_touchable():
    prices = _panel(start="2026-03-01", n_days=220)
    with pytest.raises(ConfirmationLeakError):
        feat_mod.build_frame(prices)
    dev_only = _panel()
    frame = feat_mod.build_frame(dev_only)
    assert periods.check_no_confirmation(frame["t0"], "test") is None
