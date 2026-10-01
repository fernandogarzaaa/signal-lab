"""Phase 3 benchmark tests.

Covers the go/no-go gate: the pre-specified verdict rule, the chance-level
tripwire (shuffled labels), the leakage-sensitivity tripwire (an injected
future-leaking price feature must inflate variant A), fold-index identity
with the validation framework, train-only fitting of the scaler/imputer,
determinism, and the calibration/abstention diagnostics.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from signal_lab.benchmark.phase3 import (
    PRIMARY_MODEL,
    PRICE_COLS,
    BenchmarkConfig,
    _build_matrices,
    _ece,
    _paired_test,
    fold_metrics,
    gate_verdict,
    run_benchmark,
)
from signal_lab.models.labels import LabelConfig
from signal_lab.validation import periods, splits


def _synthetic_df(n=120, seed=0, leak=False):
    rng = np.random.default_rng(seed)
    tickers = ["AAA", "BBB", "CCC"]
    base = date(2025, 2, 3)  # a Monday; all rows are development period
    rows = []
    for i in range(n):
        d = base + timedelta(days=i // 3)
        abn = float(rng.normal(0, 0.03))
        rows.append(
            {
                "ticker": tickers[i % 3],
                "pub_date": d,
                "published_at": pd.Timestamp(d) + pd.Timedelta(hours=8),
                "ctx_asof": pd.Timestamp(d) - pd.Timedelta(days=1),
                "t0": pd.Timestamp(d),
                "t1": pd.Timestamp(d) + pd.Timedelta(days=3),
                "abn_ret": abn,
                "excess_3d": abn,
                "direction_3d": 1.0 if abn > 0 else 0.0,
                "text": " ".join(
                    rng.choice(
                        ["market", "stock", "price", "trade", "news", "report",
                         "earnings", "growth"],
                        size=12,
                    )
                ),
            }
        )
    df = pd.DataFrame(rows)
    for c in PRICE_COLS:
        df[c] = rng.normal(0, 1, n)
    if leak:
        # Future-leaking price feature: the full-sample top-decile flag.
        # Mirrors what a leaked cutoff/target would give the model.
        q90 = df["abn_ret"].quantile(0.90)
        df[PRICE_COLS[0]] = (df["abn_ret"] > q90).astype(float)
    return df


def _trading_days(df):
    days = pd.to_datetime(df["t0"]).dt.date.apply(lambda d: d.toordinal())
    return np.sort(days.unique().astype(np.int64))


_fb_cache_dirs: dict[tuple[str, ...], str] = {}


def _fb_cache(texts) -> str:
    """Fake FinBERT cache dir with synthetic 768-dim vectors for `texts`.

    The benchmark's FinBERT arm is cache-only by design, so tests populate
    a throwaway cache keyed exactly like the real one (no torch needed).
    """
    import tempfile

    from signal_lab.models.embeddings import _cache_stem, cache_key

    key = tuple(sorted({str(t) for t in texts}))
    if key not in _fb_cache_dirs:
        d = tempfile.mkdtemp(prefix="fb-test-")
        rng = np.random.default_rng(1234)
        for t in key:
            vec = rng.normal(0, 1, 768).astype(np.float32)
            np.save(f"{d}/{_cache_stem(cache_key(t))}.npy", vec)
        _fb_cache_dirs[key] = d
    return _fb_cache_dirs[key]


def _fast_config(**kw):
    base = dict(
        n_splits=2,
        min_train=20,
        calibrate=False,
        abstain=False,
        model_names=("naive", "historical_rate", "logreg_balanced"),
    )
    base.update(kw)
    return BenchmarkConfig(**base).validated()


def _no_attention():
    return LabelConfig(attention_enabled=False).validated()


def test_gate_verdict_go():
    paired = {
        PRIMARY_MODEL: {
            "C-A/pr_auc": _paired_test(
                [0.13, 0.14, 0.12, 0.15, 0.13],
                [0.10, 0.10, 0.10, 0.10, 0.10],
            )
        }
    }
    g = gate_verdict(paired)
    assert g["verdict"] == "GO"
    assert g["primary_mean_diff"] > 0
    assert g["primary_ci95"][0] > 0
    assert g["test_period_touched"] is False


def test_gate_verdict_nogo_inconclusive():
    paired = {
        PRIMARY_MODEL: {
            "C-A/pr_auc": _paired_test(
                [0.11, 0.09, 0.12, 0.10, 0.11],
                [0.10, 0.10, 0.10, 0.10, 0.10],
            )
        }
    }
    g = gate_verdict(paired)
    assert g["verdict"] == "NO-GO"


def test_gate_verdict_nogo_negative():
    paired = {
        PRIMARY_MODEL: {
            "C-A/pr_auc": _paired_test(
                [0.09, 0.08, 0.09, 0.07, 0.09],
                [0.10, 0.10, 0.10, 0.10, 0.10],
            )
        }
    }
    g = gate_verdict(paired)
    assert g["verdict"] == "NO-GO"
    assert g["primary_mean_diff"] < 0


def test_ece_known_value():
    y = np.array([0, 1])
    p = np.array([0.25, 0.75])
    assert _ece(y, p, n_bins=2) == pytest.approx(0.25)


def test_fold_metrics_constant_proba():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    p = np.full(10, 0.5)
    m = fold_metrics(y, p, excess=y.astype(float), direction=y.astype(float))
    assert m["pr_auc"] == pytest.approx(0.3)  # constant score -> pos rate
    assert m["roc_auc"] == pytest.approx(0.5)
    assert np.isnan(m["rank_ic"])
    assert m["baseline_pr_auc"] == pytest.approx(0.3)


def test_trading_days_required():
    df = _synthetic_df()
    with pytest.raises(ValueError, match="trading_days is required"):
        run_benchmark(df, None, label_cfg=_no_attention(), config=_fast_config())


def test_benchmark_runs_and_gate_present():
    df = _synthetic_df()
    res = run_benchmark(
        df, _trading_days(df), label_cfg=_no_attention(), config=_fast_config(),
        finbert_cache_dir=_fb_cache(df["text"]),
    )
    assert res["phase"] == 3
    assert res["n_folds_completed"] == 2
    assert res["gate"]["verdict"] in ("GO", "NO-GO")
    assert res["gate"]["test_period_touched"] is False
    for variant in ("A", "B", "C", "Fb", "Fc"):
        assert "logreg_balanced" in res["aggregate"][variant]
        row = res["aggregate"][variant]["logreg_balanced"]["pr_auc"]
        assert 0.0 <= row["mean"] <= 1.0
    # naive model must sit exactly at the chance level per fold
    for f in res["folds"]:
        for variant in ("A", "B", "C", "Fb", "Fc"):
            m = f["metrics"][variant]["naive"]
            assert m["pr_auc"] == pytest.approx(m["baseline_pr_auc"])


def test_shuffled_labels_near_chance():
    """Tripwire: with the label signal destroyed, no model may beat chance
    by a wide margin."""
    df = _synthetic_df(seed=3)
    df = df.copy()
    df["abn_ret"] = np.random.default_rng(99).permutation(df["abn_ret"].to_numpy())
    df["excess_3d"] = df["abn_ret"]
    df["direction_3d"] = (df["abn_ret"] > 0).astype(float)
    res = run_benchmark(
        df, _trading_days(df), label_cfg=_no_attention(), config=_fast_config(),
        finbert_cache_dir=_fb_cache(df["text"]),
    )
    for f in res["folds"]:
        base = f["metrics"]["C"]["naive"]["baseline_pr_auc"]
        got = f["metrics"]["C"]["logreg_balanced"]["pr_auc"]
        assert abs(got - base) < 0.25, f"fold {f['fold']}: {got} vs chance {base}"


def test_future_leak_detected():
    """Tripwire: an injected future-leaking price feature must inflate
    variant A far above chance, proving the harness can see leakage."""
    df = _synthetic_df(seed=5, leak=True)
    res = run_benchmark(
        df, _trading_days(df), label_cfg=_no_attention(), config=_fast_config(),
        finbert_cache_dir=_fb_cache(df["text"]),
    )
    a = res["aggregate"]["A"]["logreg_balanced"]["pr_auc"]["mean"]
    assert a > 0.7, f"leak not detected: variant A pr_auc={a}"


def test_fold_test_indices_match_validation_framework():
    """Variants A/B/C share identical test indices, and those indices are
    exactly the ones splits.make_splits yields (the run_walk_forward
    convention)."""
    df = _synthetic_df()
    res = run_benchmark(
        df, _trading_days(df), label_cfg=_no_attention(), config=_fast_config(),
        finbert_cache_dir=_fb_cache(df["text"]),
    )
    work = df.dropna(subset=["published_at", "t0", "t1", "abn_ret"]).reset_index(
        drop=True
    )
    work = work[
        pd.to_datetime(work["t0"]).dt.date <= periods.DEV_END
    ].reset_index(drop=True)
    wf_cfg = splits.WalkForwardConfig(
        n_splits=2, window="expanding", embargo_days=5, horizon_days=3, min_train=20
    )
    expected = [list(sp["test_idx"]) for sp in splits.make_splits(
        work, wf_cfg, _trading_days(df))]
    got = [f["test_idx"] for f in res["folds"]]
    assert got == expected


def test_scaler_imputer_fit_on_train_only():
    """The median imputer and scaler must use train statistics even when the
    test block contains an extreme outlier."""
    rng = np.random.default_rng(0)
    n_tr, n_te = 40, 10
    dtr = pd.DataFrame(
        {c: rng.normal(0, 1, n_tr) for c in PRICE_COLS}
    )
    dte = pd.DataFrame(
        {c: rng.normal(0, 1, n_te) for c in PRICE_COLS}
    )
    dte.iloc[0, 0] = 1e6  # extreme test-only outlier
    dte.iloc[1, 3] = np.nan  # test NaN must be imputed, not crash
    texts_tr = pd.Series(["market news report"] * n_tr)
    texts_te = pd.Series(["market news report"] * n_te)
    mats_tr, mats_te, _ = _build_matrices(
        texts_tr, texts_te, dtr, dte,
        finbert_cache_dir=_fb_cache(list(texts_tr) + list(texts_te)),
    )
    got = mats_te["A"].toarray()
    assert np.isfinite(got).all()
    # column 0, row 0: (1e6 - train_median) / train_std
    col = dtr.iloc[:, 0].to_numpy()
    med = np.median(col)
    std = col.std()  # StandardScaler uses biased (ddof=0) std
    assert got[0, 0] == pytest.approx((1e6 - med) / std, rel=1e-6)


def test_determinism():
    df = _synthetic_df(seed=7)
    kw = dict(label_cfg=_no_attention(), config=_fast_config())
    kw["finbert_cache_dir"] = _fb_cache(df["text"])
    r1 = run_benchmark(df, _trading_days(df), **kw)
    r2 = run_benchmark(df, _trading_days(df), **kw)
    for v in ("A", "B", "C", "Fb", "Fc"):
        a = r1["aggregate"][v]["logreg_balanced"]["pr_auc"]["mean"]
        b = r2["aggregate"][v]["logreg_balanced"]["pr_auc"]["mean"]
        assert a == b


def test_calibration_and_abstention_diagnostics():
    cfg = BenchmarkConfig(
        n_splits=2,
        min_train=20,
        calibrate=True,
        abstain=True,
        min_calib_train=20,
        model_names=("logreg_balanced",),
    ).validated()
    df = _synthetic_df()
    res = run_benchmark(df, _trading_days(df), label_cfg=_no_attention(), config=cfg,
                        finbert_cache_dir=_fb_cache(df["text"]))
    f0 = res["folds"][0]
    assert "calibration" in f0
    cal = f0["calibration"]
    if "skipped" in cal:
        # Degenerate fold: the skip itself must be recorded loudly.
        assert isinstance(cal["skipped"], str)
    else:
        for v in ("A", "B", "C"):
            row = cal[v]
            assert "raw" in row
            for k in ("sigmoid", "isotonic"):
                assert k in row
                if "skipped" in row[k]:
                    continue
                assert 0.0 <= row[k]["brier"] <= 1.0
                assert 0.0 <= row[k]["ece"] <= 1.0
                assert row[k]["proba_std"] >= 0.0
    assert "abstention" in f0
    assert f0["abstention"]["full"]["coverage"] == 1.0


def test_calibration_skips_single_class_cv_partition():
    """A TimeSeriesSplit(3) internal train partition that is single-class
    must skip calibration loudly, not crash the benchmark."""
    from scipy.sparse import csr_matrix

    from signal_lab.benchmark.phase3 import _calibration_diagnostic

    rng = np.random.default_rng(0)
    n = 40
    # First TimeSeriesSplit(3) train partition covers rows [0:10]: all negative.
    ytr = np.ones(n, dtype=int)
    ytr[:10] = 0
    ytr[10:15] = 0
    assert len(np.unique(ytr)) == 2  # the full train block has both classes
    X = csr_matrix(rng.normal(0, 1, (n, 5)))
    yte = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    Xe = csr_matrix(rng.normal(0, 1, (8, 5)))
    mats = {"A": X, "B": X, "C": X}
    mates = {"A": Xe, "B": Xe, "C": Xe}
    out = _calibration_diagnostic(
        mats, mates, ytr, yte, None, seed=7, min_calib_train=20,
        log=lambda *a, **k: None,
    )
    assert out == {"skipped": "single_class_cv_partition"}
