"""Leakage tests: the canary, the shuffle, and the mechanism.

These are the tests that prove the validation framework is honest:

1. CANARY (test_canary_caught_by_purge): a "future feature" is injected:
   rows in the purge zone (t1 reaches into the test period) carry text
   that reveals their label. WITHOUT purge the model scores PR-AUC 1.0
   (proving the canary is a real leak); WITH purge the same setup drops
   to 0.5 (chance on the balanced test set), proving the purge removes
   the leak. The purge escape hatch exists ONLY for this test.

2. SHUFFLE (test_shuffled_labels_score_chance): with abn_ret shuffled
   (no text-label relationship), PR-AUC must be chance-level. If the
   pipeline leaked label information through any channel (features,
   cutoff, attention, vectorizer), this would inflate.

3. MECHANISM (test_purge_removes_exactly_the_canary_rows): the purge
   removes precisely the rows whose [t0, t1] reaches into the test
   period: no more, no fewer.
"""

import numpy as np
import pandas as pd

from signal_lab.models.labels import LabelConfig
from signal_lab.models import walk_forward as wf
from signal_lab.validation import splits as S
from signal_lab.validation.labels import fold_cutoff, fold_labels

N_TICKERS = 5
N_DAYS = 120
# Fold 2 of a 2-fold split on 120 days x 5 tickers: blocks of 40 days,
# test starts day 80, purge zone = days 77-79 (t1 = t0 + 3 >= 80).
PURGE_ZONE = range(77, 80)
TEST_START_DAY = 80


def _days():
    return pd.bdate_range("2026-01-05", periods=N_DAYS + 3)


def _frame(seed=0, inject_canary=True, shuffle_abn=False):
    """Canary frame: days 0-76 normal; days 77-79 (purge zone) carry the
    injected future text; days 80+ balanced with text revealing labels."""
    rng = np.random.default_rng(seed)
    days = _days()
    d = [x.date() for x in days[:N_DAYS]]
    rows = []
    for ti in range(N_TICKERS):
        for i in range(N_DAYS):
            if inject_canary and i in PURGE_ZONE:
                text, abn = "canary bull signal", 0.10
            elif i >= TEST_START_DAY:
                pos = i % 2 == 0
                text = "canary bull signal" if pos else "canary bear signal"
                abn = 0.10 if pos else -0.10
            else:
                text, abn = "normal market news update", rng.normal(0, 0.02)
            rows.append(
                {
                    "published_at": pd.to_datetime(
                        days[i] + pd.Timedelta(hours=15), utc=True
                    ),
                    "pub_date": d[i],
                    "ticker": f"T{ti}",
                    "text": text,
                    "t0": d[i],
                    "t1": days[i + 3].date(),
                    "abn_ret": abn,
                    "ctx_asof": (days[i] - pd.offsets.BDay(1)).date(),
                }
            )
    df = pd.DataFrame(rows)
    if shuffle_abn:
        df["abn_ret"] = rng.permutation(df["abn_ret"].to_numpy())
    trading_days = np.sort(
        np.array([x.date().toordinal() for x in days], dtype=np.int64)
    )
    return df, trading_days


def _cfg():
    return LabelConfig(attention_min_articles=1).validated()


def _fold2_pr_auc(df, td, purge):
    res = wf.run_walk_forward(
        df, n_splits=2, trading_days=td, label_cfg=_cfg(), min_train=20,
        purge=purge, log=lambda *a, **k: None,
    )
    assert not res["skipped_folds"], res["skipped_folds"]
    f2 = next(f for f in res["folds"] if f["fold"] == 2)
    return f2["pr_auc"]


def test_canary_caught_by_purge():
    """The canary: without purge the injected future text scores 1.0;
    with purge it drops to chance (0.5 on the balanced test set)."""
    for seed in (0, 1):
        df, td = _frame(seed=seed)
        pr_no_purge = _fold2_pr_auc(df, td, purge=False)
        pr_purge = _fold2_pr_auc(df, td, purge=True)
        assert pr_no_purge > 0.95, (seed, pr_no_purge)
        assert pr_purge < 0.60, (seed, pr_purge)


def test_shuffled_labels_score_chance():
    """Shuffled abn_ret destroys the text-label link; PR-AUC must be
    chance-level (near the ~10% test positive rate), far below the
    canary's 1.0."""
    for seed in (0, 1):
        df, td = _frame(seed=seed, inject_canary=False, shuffle_abn=True)
        pr = _fold2_pr_auc(df, td, purge=True)
        assert pr < 0.30, (seed, pr)


def test_purge_removes_exactly_the_canary_rows():
    """Mechanism: the purge removes precisely the rows whose [t0, t1]
    reaches into the test period (the canary rows), no more, no fewer."""
    df, td = _frame(seed=0)
    cfg = S.WalkForwardConfig(
        n_splits=2, horizon_days=3, embargo_days=5, min_train=20, purge=True
    )
    splits = S.make_splits(df, cfg, td)
    s2 = splits[1]
    assert s2["test_start"] == _days()[TEST_START_DAY].date().isoformat()

    # Which rows were purged? Recompute: train rows with t1 >= test_start.
    n = len(df)
    # The split used the dev-filtered frame; reconstruct the kept positions.
    # Simpler: purged count must equal (# purge-zone rows) = 3 days x 5 tickers.
    assert s2["n_purged"] == len(PURGE_ZONE) * N_TICKERS, s2["n_purged"]

    # And the kept train rows contain no canary text.
    kept_texts = df.iloc[s2["train_idx"]]["text"]
    assert not kept_texts.str.contains("canary").any()


def test_canary_labels_are_train_only():
    """The fold-2 cutoff is computed on train abn_ret only; the canary
    rows (abn_ret=0.10) must not drag it up when purged."""
    df, td = _frame(seed=0)
    cfg = S.WalkForwardConfig(
        n_splits=2, horizon_days=3, embargo_days=5, min_train=20, purge=True
    )
    s2 = S.make_splits(df, cfg, td)[1]
    train_abn = df.iloc[s2["train_idx"]]["abn_ret"]
    cutoff = fold_cutoff(train_abn, _cfg().quantile)
    # Train is ~N(0, 0.02): 0.9 quantile ~ 0.026. If the 15 canary rows
    # (abn_ret=0.10) leaked into the cutoff, it would be much higher.
    assert cutoff < 0.05, cutoff
    labels = fold_labels(train_abn, cutoff)
    assert labels.mean() < 0.15  # ~10% positive rate, not inflated
