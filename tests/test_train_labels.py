"""Tests for training-label source selection (no API calls, no DB)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signal_lab.models import train
from signal_lab.models.walk_forward import run_walk_forward
from signal_lab.models.train_labels import (
    TRAIN_LABEL_COL,
    TRAIN_LABEL_SOURCES,
    apply_train_labels,
    load_jev_labels,
)


def _jev_file(tmp_path: Path, items) -> Path:
    p = tmp_path / "jev.json"
    p.write_text(json.dumps({"items": items}))
    return p


def _jev_items():
    return [
        {"url": "https://x.test/0", "jev_label": "positive", "jev_confidence": 0.9},
        {"url": "https://x.test/1", "jev_label": "negative", "jev_confidence": 0.4},
        {"url": "https://x.test/2", "jev_label": "neutral", "jev_confidence": 0.7},
        {"url": "https://x.test/3", "jev_label": "positive", "jev_confidence": 0.55},
    ]


def _df():
    return pd.DataFrame(
        {
            "url": [f"https://x.test/{i}" for i in range(4)],
            "label": [1, 0, 1, 0],
        }
    )


def _quiet(*a, **k):
    pass


def test_sources_are_known():
    assert TRAIN_LABEL_SOURCES == ("weak", "jev", "jev-conf06")


def test_weak_returns_df_unchanged(tmp_path):
    df = _df()
    out = apply_train_labels(df, "weak", log=_quiet)
    assert TRAIN_LABEL_COL not in out.columns
    pd.testing.assert_frame_equal(out, df)


def test_unknown_source_raises():
    with pytest.raises(ValueError, match="unknown train-label source"):
        apply_train_labels(_df(), "nope", log=_quiet)


def test_jev_keeps_directionals_only(tmp_path):
    p = _jev_file(tmp_path, _jev_items())
    out = apply_train_labels(_df(), "jev", jev_path=p, log=_quiet)
    got = out[TRAIN_LABEL_COL].tolist()
    assert got[0] == 1.0  # positive
    assert got[1] == 0.0  # negative
    assert pd.isna(got[2])  # neutral -> abstain
    assert got[3] == 1.0


def test_jev_conf06_drops_low_confidence(tmp_path):
    p = _jev_file(tmp_path, _jev_items())
    out = apply_train_labels(_df(), "jev-conf06", jev_path=p, log=_quiet)
    got = out[TRAIN_LABEL_COL].tolist()
    assert got[0] == 1.0  # conf 0.9 kept
    assert pd.isna(got[1])  # conf 0.4 dropped
    assert pd.isna(got[2])  # neutral dropped
    assert pd.isna(got[3])  # conf 0.55 dropped


def test_missing_jev_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Jev labels not found"):
        load_jev_labels(tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="Jev labels not found"):
        apply_train_labels(_df(), "jev", jev_path=tmp_path / "nope.json",
                           log=_quiet)


def test_partial_coverage_raises(tmp_path):
    p = _jev_file(tmp_path, _jev_items()[:2])  # only 2 of 4 urls
    with pytest.raises(ValueError, match="no Jev label"):
        apply_train_labels(_df(), "jev", jev_path=p, log=_quiet)


def _synthetic_df(n=40, flip_train=False):
    rows = []
    for i in range(n):
        good = i % 2 == 0
        rows.append(
            {
                "url": f"https://x.test/{i}",
                "title": f"{'good' if good else 'bad'} news {i}",
                "body_snippet": "",
                "published_at": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "label": int(good),
                TRAIN_LABEL_COL: float(int(not good) if flip_train else int(good)),
            }
        )
    return pd.DataFrame(rows)


def _pr_auc(result, name="logreg_balanced"):
    return next(r.pr_auc for r in result["reports"] if r.name == name)


def _accuracy(result, name="logreg_balanced"):
    return next(r.accuracy for r in result["reports"] if r.name == name)


def test_train_label_col_identical_to_label_matches_default():
    df = _synthetic_df()
    base = train(df, text_col="title")
    same = train(df, text_col="title", train_label_col=TRAIN_LABEL_COL)
    assert _pr_auc(base) == pytest.approx(_pr_auc(same))
    assert base["n_train"] == same["n_train"]
    assert base["n_test"] == same["n_test"]


def test_train_label_col_abstentions_shrink_train_not_test():
    df = _synthetic_df()
    df.loc[df.index[:10], TRAIN_LABEL_COL] = float("nan")
    base = train(df, text_col="title")
    abst = train(df, text_col="title", train_label_col=TRAIN_LABEL_COL)
    assert abst["n_train"] < base["n_train"]
    assert abst["n_test"] == base["n_test"]


def test_test_metrics_use_weak_labels_not_train_labels():
    # Train labels are flipped relative to weak labels. Test accuracy
    # against the weak labels must be ~0 for the flipped model (it would
    # be ~1 if the implementation wrongly scored against train labels).
    df = _synthetic_df(flip_train=True)
    flipped = train(df, text_col="title", train_label_col=TRAIN_LABEL_COL)
    correct = train(df, text_col="title")
    assert _accuracy(flipped) < 0.5
    assert _accuracy(correct) > 0.5


def test_train_label_col_all_nan_raises():
    df = _synthetic_df()
    df[TRAIN_LABEL_COL] = float("nan")
    with pytest.raises(ValueError, match="no usable training rows"):
        train(df, text_col="title", train_label_col=TRAIN_LABEL_COL)


def test_train_label_col_missing_column_raises():
    df = _synthetic_df().drop(columns=[TRAIN_LABEL_COL])
    with pytest.raises(ValueError, match="not in df columns"):
        train(df, text_col="title", train_label_col=TRAIN_LABEL_COL)


def _wf_df(n=60, abstain_every=0, flip_train=False):
    """Synthetic frame for the new run_walk_forward: weak labels come from
    abn_ret via the per-fold cutoff; TRAIN_LABEL_COL carries alternate
    (possibly flipped) train labels. Text aligns with abn_ret so a model
    trained on consistent labels scores well."""
    from signal_lab.models.labels import LabelConfig  # local: keeps import light

    days = pd.bdate_range("2026-01-05", periods=n + 3)
    d_dates = [d.date() for d in days[:n]]
    rows = []
    for i in range(n):
        good = i % 2 == 0
        tl = int(not good) if flip_train else int(good)
        train_label = float("nan") if abstain_every and i % abstain_every == 0 else float(tl)
        rows.append(
            {
                "title": f"{'good' if good else 'bad'} news {i}",
                "body_snippet": "",
                "published_at": (days[i] + pd.Timedelta(hours=15)).tz_localize("UTC"),
                "pub_date": d_dates[i],
                "ticker": "AAA",
                "t0": d_dates[i],
                "t1": days[i + 3].date(),
                "abn_ret": 0.05 if good else -0.05,
                "ctx_asof": (days[i] - pd.offsets.BDay(1)).date(),
                TRAIN_LABEL_COL: train_label,
            }
        )
    df = pd.DataFrame(rows)
    trading_days = np.sort(
        np.array([d.date().toordinal() for d in days], dtype=np.int64)
    )
    return df, trading_days


def _wf_kwargs(df, trading_days, **kw):
    from signal_lab.models.labels import LabelConfig

    base = dict(
        n_splits=2,
        trading_days=trading_days,
        label_cfg=LabelConfig(attention_min_articles=1).validated(),
        min_train=5,
        log=lambda *a, **k: None,
    )
    base.update(kw)
    return base


def test_walk_forward_train_labels_swap():
    # Flipped training labels must hurt the weak-label test metrics,
    # proving the fold loop trains on the alternate labels.
    df_f, td_f = _wf_df(flip_train=True)
    df_c, td_c = _wf_df()
    flipped = run_walk_forward(
        df_f, train_label_col=TRAIN_LABEL_COL, **_wf_kwargs(df_f, td_f),
    )
    correct = run_walk_forward(df_c, **_wf_kwargs(df_c, td_c))
    f_mean = flipped["aggregate"]["pr_auc"]["mean"]
    c_mean = correct["aggregate"]["pr_auc"]["mean"]
    assert f_mean < 0.6 < c_mean, (f_mean, c_mean)


def test_walk_forward_abstentions_excluded_from_train_only():
    df, td = _wf_df(abstain_every=3)
    res = run_walk_forward(
        df, train_label_col=TRAIN_LABEL_COL, **_wf_kwargs(df, td),
    )
    assert res["folds"], "expected at least one scored fold"
    for f in res["folds"]:
        # every third train row abstained -> train shrinks, test untouched
        assert f["n_train"] < f["n_test"] * 2
    assert res["train_labels"]["test_labels"].startswith("weak")


def test_walk_forward_missing_train_label_col_raises():
    df, td = _wf_df()
    with pytest.raises(ValueError, match="not in df columns"):
        run_walk_forward(df, train_label_col="nope", **_wf_kwargs(df, td))


def test_walk_forward_records_train_label_source():
    df, td = _wf_df()
    res = run_walk_forward(
        df, train_label_col=TRAIN_LABEL_COL, train_label_source="jev-conf06",
        **_wf_kwargs(df, td),
    )
    assert res["train_labels"]["source"] == "jev-conf06"
    res_weak = run_walk_forward(df, **_wf_kwargs(df, td))
    assert res_weak["train_labels"]["source"] == "weak"
