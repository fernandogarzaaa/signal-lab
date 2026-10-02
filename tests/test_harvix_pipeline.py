"""End-to-end HARVIX pipeline test on synthetic data (no network).

Builds a small price panel, the frozen frame, HAR features, synthetic
VIX features, drops NaN rows, runs a small purged walk-forward with the
frozen ridge arms, scores QLIKE, computes the verdict, and smoke-tests
the MCS and DSR on the pooled losses. Also verifies the confirmation
period is never touchable.
"""

import numpy as np
import pandas as pd
import pytest

from signal_lab.eventvol import mcs as mcs_mod
from signal_lab.harvix import features as feat_mod
from signal_lab.harvix import run_campaign as rc
from signal_lab.stats import deflated_sharpe as dsr_mod
from signal_lab.validation import periods
from signal_lab.validation.periods import ConfirmationLeakError
from signal_lab.validation.splits import WalkForwardConfig, make_splits
from signal_lab.vol import features as vol_feat
from signal_lab.vol import frame as frame_mod
from signal_lab.vol import metrics as metrics_mod
from signal_lab.vol import models as models_mod


def _panel(tickers=("AAA", "BBB", "SPY"), n_days=130, start="2022-01-03",
           seed=21):
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


def _vix(n_days=130, start="2022-01-03", seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    vix = np.clip(20.0 + np.cumsum(rng.normal(0, 0.5, n_days)), 5.0, None)
    vix3m = vix * (1.0 + rng.normal(0.05, 0.01, n_days))
    return pd.DataFrame({"date": dates, "vix": vix, "vix3m": vix3m})


def _clean_frame():
    prices = _panel()
    frame = frame_mod.build_frame(prices)
    frame = vol_feat.add_features(frame, prices)
    frame = feat_mod.add_vix_features(frame, _vix())
    clean, n_dropped = vol_feat.drop_nan_features(
        frame, feat_mod.HARVIX_COLUMNS, log=lambda m: None)
    assert n_dropped > 0  # burn-in rows really drop
    assert len(clean) > 100
    return clean, prices


def _calendar(prices):
    cal = frame_mod.trading_calendar(prices)
    return np.array(sorted(d.date().toordinal() for d in cal), dtype=np.int64)


def test_end_to_end_walk_forward_verdict():
    clean, prices = _clean_frame()
    config = WalkForwardConfig(n_splits=2, window="expanding", horizon_days=5,
                               embargo_days=5, min_train=10, seed=7)
    splits = make_splits(clean, config, _calendar(prices))
    assert len(splits) == 2

    per_fold_d = []
    losses, t0s = [], []
    for s in splits:
        y = clean["target"]
        Xhv = clean[feat_mod.HARVIX_COLUMNS]
        Xh = clean[feat_mod.HAR_COLUMNS]
        fh = models_mod.fit_ridge(Xhv.iloc[s["train_idx"]], y.iloc[s["train_idx"]])
        fh0 = models_mod.fit_ridge(Xh.iloc[s["train_idx"]], y.iloc[s["train_idx"]])
        fc_hv = models_mod.predict_ridge(fh, Xhv.iloc[s["test_idx"]])
        fc_h = models_mod.predict_ridge(fh0, Xh.iloc[s["test_idx"]])
        yt = y.iloc[s["test_idx"]].to_numpy()
        q_hv = metrics_mod.mean_qlike(yt ** 2, fc_hv ** 2)
        q_h = metrics_mod.mean_qlike(yt ** 2, fc_h ** 2)
        per_fold_d.append(q_h - q_hv)
        losses.append(pd.DataFrame({
            "har_vix": metrics_mod.qlike(yt ** 2, fc_hv ** 2),
            "har": metrics_mod.qlike(yt ** 2, fc_h ** 2),
        }))
        t0s.append(pd.to_datetime(clean.iloc[s["test_idx"]]["t0"])
                   .reset_index(drop=True))

    d = np.array(per_fold_d)
    ci = metrics_mod.t_confidence_interval(d, alpha=0.05)
    dm = metrics_mod.diebold_mariano(d)
    verdict = "GO" if (ci["mean"] > 0.0 and ci["lower"] > 0.0) else "NO-GO"
    assert verdict in ("GO", "NO-GO")
    assert np.isfinite(dm["statistic"]) and 0.0 <= dm["p_value"] <= 1.0

    # MCS smoke on the pooled per-row losses.
    loss_df = pd.concat(losses, ignore_index=True)
    t0_all = pd.concat(t0s, ignore_index=True)
    mcs_out = mcs_mod.mcs(loss_df, t0_all, n_boot=200, mean_block=5,
                          log=lambda m: None)
    assert set(mcs_out["survivors"]) <= {"har_vix", "har"}
    assert mcs_out["n_rows"] == len(loss_df)

    # DSR smoke on the pooled gains (non-degenerate pseudo-returns).
    rng = np.random.default_rng(3)
    n = len(loss_df)
    gains = {"har_vix": rng.normal(0.01, 0.05, n),
             "har": rng.normal(0.005, 0.05, n)}
    report = dsr_mod.dsr_report(gains, freq=1)
    assert set(report["variant"]) == {"har_vix", "har"}


def test_harvix_feat_drop_nan_uses_34_columns():
    clean, _ = _clean_frame()
    assert list(clean.columns[-3:]) == feat_mod.VIX_FEATURE_COLUMNS
    assert clean[feat_mod.HARVIX_COLUMNS].notna().all().all()


def test_confirmation_frame_build_raises():
    prices = _panel(start="2026-09-01", n_days=40)
    with pytest.raises(ConfirmationLeakError):
        frame_mod.build_frame(prices)


def test_confirmation_rows_rejected_by_splits():
    df = pd.DataFrame({
        "t0": [pd.Timestamp("2026-09-02").date()] * 4,
        "t1": [pd.Timestamp("2026-09-09").date()] * 4,
        "published_at": [pd.Timestamp("2026-09-02", tz="UTC")] * 4,
        "target": [0.01] * 4,
    })
    config = WalkForwardConfig(n_splits=2, window="expanding", horizon_days=5,
                               embargo_days=5, min_train=1, seed=7)
    trading = np.array([pd.Timestamp("2026-09-02").date().toordinal()],
                       dtype=np.int64)
    with pytest.raises(ConfirmationLeakError):
        make_splits(df, config, trading)


def test_dev_end_boundary():
    assert periods.DEV_END.isoformat() == "2026-06-30"
