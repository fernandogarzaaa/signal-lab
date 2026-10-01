"""Tests for validation.labels: train-only cutoff and attention filtering.

The audit found the label cutoff and attention quantiles were computed on
the FULL sample before splitting (findings 4c, 4d). These tests pin the
fix: both statistics are computed on train rows only, and the test rows'
labels/keep-mask are derived by applying the frozen train statistics.
"""

import numpy as np
import pandas as pd

from signal_lab.models.labels import LabelConfig
from signal_lab.validation.labels import (
    attention_keep_mask,
    fold_cutoff,
    fold_labels,
)


def _cfg(**kw):
    base = dict(window_days=3, quantile=0.9)
    base.update(kw)
    return LabelConfig(**base).validated()


def _days(n, start="2026-01-05"):
    return pd.bdate_range(start, periods=n)


def test_cutoff_uses_train_only():
    # Train abn_ret ~ N(0, 1); the test period is drawn from N(3, 1)
    # (a hot regime). A full-sample cutoff would be dragged up by the
    # test regime; the train-only cutoff must ignore it.
    rng = np.random.default_rng(0)
    train_abn = pd.Series(rng.normal(0, 1, 500))
    test_abn = pd.Series(rng.normal(3, 1, 100))
    cfg = _cfg()
    assert fold_cutoff(train_abn, cfg.quantile) == np.quantile(train_abn, 0.9)
    # Full-sample cutoff is much higher (the test regime leaks in).
    assert np.quantile(pd.concat([train_abn, test_abn]), 0.9) > fold_cutoff(
        train_abn, cfg.quantile
    ) + 1.0


def test_fold_labels_apply_frozen_cutoff():
    cfg = _cfg()
    train_abn = pd.Series([0.01, 0.02, 0.03, 0.04, 0.10])
    cutoff = fold_cutoff(train_abn, cfg.quantile)
    # 0.9 quantile of 5 values: linear interpolation between 0.04 and 0.10.
    assert cutoff == np.quantile(train_abn, 0.9)
    labels = fold_labels(pd.Series([0.0, 0.05, 0.20]), cutoff)
    assert labels.tolist() == [0, int(0.05 >= cutoff), 1]


def test_attention_quantiles_from_train_only():
    # Train ticker-days each have 4 articles; the test period has five
    # days with 5 articles each. Train-only: qrank(5) = 1.0 -> kept.
    # The OLD bug (full-sample distribution [4,4,4,4,5,5,5,5,5]) would
    # give qrank(5) = 4/9 < 0.75 -> dropped: test rows' keep/drop
    # depending on other test rows is the leak. min_articles=100 so only
    # the quartile branch matters.
    days = _days(9)
    train = pd.DataFrame(
        {
            "ticker": "AAA",
            "pub_date": [d.date() for d in days[:4] for _ in range(4)],
        }
    )
    test = pd.DataFrame(
        {
            "ticker": "AAA",
            "pub_date": [d.date() for d in days[4:] for _ in range(5)],
        }
    )
    mask = attention_keep_mask(train, test, min_articles=100, top_quartile=True)
    assert mask.dtype == bool
    assert mask.all(), "train-only quantiles must keep these busy test days"

    # Document the bug: full-sample quantiles would drop them.
    full_counts = np.array([4, 4, 4, 4, 5, 5, 5, 5, 5])
    assert (full_counts < 5).mean() < 0.75


def test_attention_min_articles_floor():
    days = _days(6)
    train = pd.DataFrame(
        {
            "ticker": "AAA",
            "pub_date": [days[0].date()] * 3 + [days[1].date()] * 5,
        }
    )
    test = pd.DataFrame({"ticker": "AAA", "pub_date": [days[2].date()] * 4})
    mask = attention_keep_mask(train, test, min_articles=3, top_quartile=True)
    # 4 >= min_articles=3 -> kept regardless of quantile.
    assert mask.all()


def test_attention_top_quartile_keeps_busy_days():
    # Train day-counts: 1,1,1,1,2,2,3,10 -> qrank(3) = 6/8 = 0.75 -> keep.
    days = _days(8)
    counts = [1, 1, 1, 1, 2, 2, 3, 10]
    train = pd.DataFrame(
        {
            "ticker": "AAA",
            "pub_date": [d.date() for d, c in zip(days, counts) for _ in range(c)],
        }
    )
    test = pd.DataFrame(
        {
            "ticker": "AAA",
            "pub_date": [days[0].date()] * 3 + [days[1].date()] * 1,
        }
    )
    # min_articles=100: the floor never binds, only the quartile matters.
    mask = attention_keep_mask(train, test, min_articles=100, top_quartile=True)
    assert mask.iloc[:3].all()  # 3 articles: qrank 0.75 >= 0.75 -> keep
    assert not mask.iloc[3]     # 1 article: qrank 0.0 -> drop


def test_attention_top_quartile_disabled():
    # With top_quartile=False, only the min_articles floor applies.
    days = _days(8)
    counts = [1, 1, 1, 1, 2, 2, 3, 10]
    train = pd.DataFrame(
        {
            "ticker": "AAA",
            "pub_date": [d.date() for d, c in zip(days, counts) for _ in range(c)],
        }
    )
    test = pd.DataFrame({"ticker": "AAA", "pub_date": [days[0].date()] * 3})
    mask = attention_keep_mask(train, test, min_articles=100, top_quartile=False)
    assert not mask.any()
