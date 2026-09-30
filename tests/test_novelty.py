"""Tests for news novelty features. No DB, no network, no torch.

The leakage tests build toy timelines with synthetic embeddings and
assert the core invariant: an article's novelty score depends only on
articles published strictly before it. Adding a future article must not
change any earlier score.
"""

import numpy as np
import pandas as pd
import pytest

from signal_lab.models.embeddings import _cache_stem, cache_key
from signal_lab.models.novelty import (
    NOVELTY_FEATURE_NAMES,
    build_novelty_features,
    compute_novelty_scores,
)


def _e1():
    v = np.zeros(768, dtype=np.float32)
    v[0] = 1.0
    return v


def _e2():
    v = np.zeros(768, dtype=np.float32)
    v[1] = 1.0
    return v


def _timeline(extra_names=()):
    """Toy timeline (all UTC).

    A: AAA 2026-01-01, emb e1
    G: AAA 2026-01-01 (same timestamp as A), emb e2 (orthogonal)
    B: AAA 2026-01-02, emb e1 (duplicate of A -> novelty ~0)
    E: BBB 2026-01-02, emb e1 (no same-ticker priors -> 1.0)
    D: AAA 2026-01-03, emb e1 (future relative to B)
    F: AAA 2026-01-04, emb None (missing -> NaN at the pure layer)
    C: AAA 2026-01-11, emb e1 (A/B/D outside the 7d window -> 1.0)
    """
    names = ["A", "G", "B", "E", "D", "F", "C"]
    tickers = ["AAA", "AAA", "AAA", "BBB", "AAA", "AAA", "AAA"]
    times = [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
        "2026-01-02T00:00:00Z",
        "2026-01-03T00:00:00Z",
        "2026-01-04T00:00:00Z",
        "2026-01-11T00:00:00Z",
    ]
    embs = [_e1(), _e2(), _e1(), _e1(), _e1(), None, _e1()]
    # `extra_names` selects a subset for the leakage tests; empty keeps all.
    sel = [i for i, n in enumerate(names) if not extra_names or n in extra_names]
    return (
        [names[i] for i in sel],
        [tickers[i] for i in sel],
        [times[i] for i in sel],
        [embs[i] for i in sel],
    )


def _scores(names, tickers, times, embs, window_days=7):
    s = compute_novelty_scores(tickers, times, embs, window_days=window_days)
    return dict(zip(names, s))


def test_duplicate_of_prior_is_not_novel():
    names, tickers, times, embs = _timeline()
    s = _scores(names, tickers, times, embs)
    assert s["B"] == pytest.approx(0.0, abs=1e-6)  # duplicate of A
    assert s["D"] == pytest.approx(0.0, abs=1e-6)  # duplicate of A and B


def test_no_priors_means_maximally_novel():
    names, tickers, times, embs = _timeline()
    s = _scores(names, tickers, times, embs)
    assert s["A"] == 1.0  # first AAA article
    assert s["E"] == 1.0  # first BBB article: ticker isolation
    # C duplicates A, but A/B/D are all outside C's 7-day window.
    assert s["C"] == 1.0


def test_same_timestamp_articles_are_not_priors_of_each_other():
    # A and G share a timestamp; neither may count the other as a prior.
    names, tickers, times, embs = _timeline()
    s = _scores(names, tickers, times, embs)
    assert s["A"] == 1.0
    assert s["G"] == 1.0  # e2 orthogonal to e1, but A isn't a prior anyway


def test_missing_embedding_is_nan_at_pure_layer():
    names, tickers, times, embs = _timeline()
    s = _scores(names, tickers, times, embs)
    assert np.isnan(s["F"])
    # F's missing embedding also excludes it as a prior: C (2026-01-11)
    # sees F (2026-01-04) inside its window but has no vector for it.
    assert s["C"] == 1.0


def test_future_article_does_not_change_earlier_scores():
    full = _timeline()
    s_full = _scores(*full)
    # Drop D (2026-01-03): B (2026-01-02) must be unaffected.
    sub = _timeline(extra_names=("A", "G", "B", "E", "F", "C"))
    s_sub = _scores(*sub)
    assert s_sub["B"] == s_full["B"]
    assert s_sub["A"] == s_full["A"]


def test_no_future_leakage_at_every_cutoff():
    """For every cutoff time, scores of articles at/before the cutoff are
    identical whether or not later articles exist in the input."""
    names, tickers, times, embs = _timeline()
    s_full = _scores(names, tickers, times, embs)
    t = pd.to_datetime(times, utc=True)
    for cutoff in sorted(set(t)):
        sel = [i for i, ti in enumerate(t) if ti <= cutoff]
        s_cut = _scores(
            [names[i] for i in sel],
            [tickers[i] for i in sel],
            [times[i] for i in sel],
            [embs[i] for i in sel],
        )
        for i in sel:
            a, b = s_cut[names[i]], s_full[names[i]]
            if np.isnan(a) or np.isnan(b):
                assert np.isnan(a) and np.isnan(b)
            else:
                assert a == b, f"{names[i]} changed when later articles removed"


def test_window_boundary_inclusive_of_lower_bound():
    # Article exactly window_days before t is inside the window.
    names = ["P", "Q"]
    tickers = ["AAA", "AAA"]
    times = ["2026-01-01T00:00:00Z", "2026-01-08T00:00:00Z"]  # exactly 7d apart
    s = _scores(names, tickers, times, [_e1(), _e1()], window_days=7)
    assert s["Q"] == pytest.approx(0.0, abs=1e-6)
    s = _scores(names, tickers, times, [_e1(), _e1()], window_days=6)
    assert s["Q"] == 1.0  # 6d window excludes P


def test_misaligned_inputs_raise():
    with pytest.raises(ValueError):
        compute_novelty_scores(["AAA"], ["2026-01-01"], [_e1(), _e1()])
    with pytest.raises(ValueError):
        compute_novelty_scores(["AAA"], ["2026-01-01"], [_e1()], window_days=0)


def _write_cache(tmp_path, texts_to_vecs):
    for text, vec in texts_to_vecs.items():
        (tmp_path / f"{_cache_stem(cache_key(text))}.npy").write_bytes(
            _npy_bytes(vec)
        )


def _npy_bytes(vec):
    import io

    buf = io.BytesIO()
    np.save(buf, np.asarray(vec, dtype=np.float32))
    return buf.getvalue()


def test_build_novelty_features_end_to_end(tmp_path):
    """The real loading path (cache dir, no torch) with the 1.0 policy for
    cache misses and the sentiment interaction terms."""
    t_new = "Apple unveils record quarterly earnings beat"
    t_dup = "Apple unveils record quarterly earnings beat"  # duplicate -> ~0
    t_old = "Unrelated macro outlook piece"
    _write_cache(tmp_path, {t_new: _e1(), t_old: _e2()})
    # t_missing is NOT in the cache -> novelty 1.0 by policy.
    t_missing = "Some article nobody embedded"
    news = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "published_at": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-10T00:00:00Z",  # outside 7d window of row 0
                    "2026-01-02T00:00:00Z",  # duplicate of row 0, in window
                    "2026-01-03T00:00:00Z",  # cache miss
                ],
                utc=True,
            ),
            "text": [t_new, t_old, t_dup, t_missing],
        }
    )
    frame = build_novelty_features(
        news, cache_dir=tmp_path, log=lambda *a, **k: None
    )
    assert list(frame.columns) == NOVELTY_FEATURE_NAMES
    assert not frame.isna().any().any()
    nov = frame["nov_novelty"].to_numpy()
    assert nov[0] == 1.0  # no priors
    assert nov[1] == 1.0  # row 0 outside the 7d window
    assert nov[2] == pytest.approx(0.0, abs=1e-6)  # duplicate of row 0
    assert nov[3] == 1.0  # cache miss -> 1.0 policy
    # Interaction terms equal polarity * novelty by construction.
    from signal_lab.models.lexicon import (
        LEXICON_FEATURE_NAMES,
        lexicon_feature_matrix,
    )

    pol = lexicon_feature_matrix(news["text"].tolist())[
        :, LEXICON_FEATURE_NAMES.index("lm_polarity")
    ]
    np.testing.assert_allclose(
        frame["nov_sentiment_x_novelty"].to_numpy(), pol * nov, rtol=1e-5
    )
    np.testing.assert_allclose(
        frame["nov_abs_sentiment_x_novelty"].to_numpy(),
        np.abs(pol) * nov,
        rtol=1e-5,
    )


def test_build_novelty_features_rejects_nan():
    news = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "published_at": pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True),
            "text": ["x"],
        }
    )
    # Empty cache dir: every row misses -> all novelty 1.0, still NaN-free.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        frame = build_novelty_features(
            news, cache_dir=Path(d), log=lambda *a, **k: None
        )
    assert (frame["nov_novelty"].to_numpy() == 1.0).all()
