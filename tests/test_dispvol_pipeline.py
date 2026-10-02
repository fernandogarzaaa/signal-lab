"""End-to-end DISPVOL pipeline test on synthetic data (no network).

Builds a small price panel, the frozen frame, HAR + synthetic VIX +
dispersion features, drops NaN rows, runs a small purged
walk-forward with the frozen ridge arms (har_vix_disp vs har_vix),
scores QLIKE, computes the frozen verdict form, and smoke-tests the
MCS and DSR on the pooled losses. Also verifies the confirmation
period is never touchable.
"""

import numpy as np
import pandas as pd
import pytest

from signal_lab.dispvol import features as feat_mod
from signal_lab.dispvol import run_campaign as rc
from signal_lab.eventvol import mcs as mcs_mod
from signal_lab.harvix import features as harvix_feat
from signal_lab.stats import deflated_sharpe as dsr_mod
from signal_lab.validation import periods
from signal_lab.validation.periods import ConfirmationLeakError
from signal_lab.validation.splits import WalkForwardConfig, make_splits
from signal_lab.vol import features as vol_feat
from signal_lab.vol import frame as frame_mod
from signal_lab.vol import metrics as metrics_mod


def _panel(n_tickers=50, n_days=150, start="2022-01-03", seed=21):
    tickers = [f"T{i:02d}" for i in range(n_tickers)] + ["SPY"]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for t in tickers:
        rets = rng.normal(0.0003, 0.012, n_days)
        close = 100.0 * np.exp(np.cumsum(rets))
        vol = rng.integers(500_000, 2_000_000, n_days).astype(float)
        for i, d in enumerate(dates):
            rows.append({
                "ticker": t, "date": d,
                "open": close[i] * 0.999,
                "high": close[i] * (1.0 + abs(rets[i]) / 2 + 0.001),
                "low": close[i] * (1.0 - abs(rets[i]) / 2 - 0.001),
                "close": close[i], "adj_close": close[i], "volume": vol[i],
            })
    return pd.DataFrame(rows)


def _vix(n_days=150, start="2022-01-03", seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    vix = np.clip(20.0 + np.cumsum(rng.normal(0, 0.5, n_days)), 5.0, None)
    vix3m = vix * (1.0 + rng.normal(0.05, 0.01, n_days))
    return pd.DataFrame({"date": dates, "vix": vix, "vix3m": vix3m})


def _built_frame():
    prices = _panel()
    frame = frame_mod.build_frame(prices)
    frame = vol_feat.add_features(frame, prices)
    frame = harvix_feat.add_vix_features(frame, _vix())
    frame = feat_mod.add_dispersion_features(frame, prices)
    return frame, prices


def test_full_feature_stack_and_nan_drop():
    frame, _ = _built_frame()
    assert set(feat_mod.CHALLENGER_COLUMNS) <= set(frame.columns)
    clean, n_dropped = rc.dispvol_feat_drop_nan(frame, log=lambda *a: None)
    assert n_dropped > 0  # dispersion 65-day burn-in drops early rows
    assert len(clean) > 0
    assert clean[feat_mod.CHALLENGER_COLUMNS].notna().all().all()


def test_walk_forward_ridge_arms_and_verdict():
    frame, prices = _built_frame()
    clean, _ = rc.dispvol_feat_drop_nan(frame, log=lambda *a: None)
    dev = frame_mod.dev_frame(clean)
    trading_days = frame_mod.union_trading_ordinals(prices)
    config = WalkForwardConfig(n_splits=2, window="expanding",
                               horizon_days=5, embargo_days=5,
                               min_train=20, seed=7)
    splits = make_splits(dev, config, trading_days)
    assert len(splits) == 2

    y = clean["target"]
    per_row = []
    fold_d = []
    for s in splits:
        train_idx, test_idx = s["train_idx"], s["test_idx"]
        chall = rc._fit_predict_ridge(
            clean[feat_mod.CHALLENGER_COLUMNS].iloc[train_idx],
            y.iloc[train_idx],
            clean[feat_mod.CHALLENGER_COLUMNS].iloc[test_idx])
        base = rc._fit_predict_ridge(
            clean[feat_mod.BASELINE_COLUMNS].iloc[train_idx],
            y.iloc[train_idx],
            clean[feat_mod.BASELINE_COLUMNS].iloc[test_idx])
        yt = y.iloc[test_idx].to_numpy()
        assert np.all(np.isfinite(chall)) and np.all(chall > 0)
        assert np.all(np.isfinite(base)) and np.all(base > 0)
        q_chall = metrics_mod.mean_qlike(yt ** 2, chall ** 2)
        q_base = metrics_mod.mean_qlike(yt ** 2, base ** 2)
        fold_d.append(q_base - q_chall)
        per_row.append(pd.DataFrame({
            "har_vix_disp": metrics_mod.qlike(yt ** 2, chall ** 2),
            "har_vix": metrics_mod.qlike(yt ** 2, base ** 2),
        }))

    # Frozen verdict form: mean(d) > 0 AND 95% t CI lower bound > 0.
    d = np.array(fold_d)
    ci = metrics_mod.t_confidence_interval(d, alpha=0.05)
    dm = metrics_mod.diebold_mariano(d)
    assert set(ci) >= {"mean", "se", "lower", "upper", "df"}
    assert set(dm) >= {"statistic", "p_value", "n"}
    assert isinstance(bool(ci["mean"] > 0.0 and ci["lower"] > 0.0), bool)

    # MCS and DSR smoke on the pooled losses (reported, not gates).
    losses = pd.concat(per_row, ignore_index=True)
    t0 = pd.to_datetime(
        pd.concat([dev.iloc[s["test_idx"]]["t0"] for s in splits],
                  ignore_index=True))
    out = mcs_mod.mcs(losses, t0, log=lambda *a: None)
    assert set(out["survivors"]) | {m for m, _ in out["eliminated"]} == {
        "har_vix_disp", "har_vix"}
    rep = dsr_mod.dsr_report(
        {
            "har_vix_disp": -losses["har_vix_disp"].to_numpy(),
            "har_vix": -losses["har_vix"].to_numpy(),
        },
        freq=1)
    assert len(rep) == 2


def test_confirmation_never_touchable():
    prices = _panel(start="2026-03-01", n_days=220)
    with pytest.raises(ConfirmationLeakError):
        frame_mod.build_frame(prices)
    dev_only = _panel()
    frame = frame_mod.build_frame(dev_only)
    assert periods.check_no_confirmation(frame["t0"], "test") is None


def test_challenger_differs_from_baseline_by_design():
    frame, _ = _built_frame()
    clean, _ = rc.dispvol_feat_drop_nan(frame, log=lambda *a: None)
    assert not set(feat_mod.DISPERSION_COLUMNS) <= set(
        feat_mod.BASELINE_COLUMNS)
    assert (clean[feat_mod.DISPERSION_COLUMNS].abs().sum().sum() > 0)
