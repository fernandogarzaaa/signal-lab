"""News novelty features: open, reproducible novelty on GDELT text.

Hypothesis under test: markets react to new information, not rehashed
stories, so sentiment modulated by novelty (sentiment x novelty) should
predict abnormal returns better than sentiment alone.

Positioning (stated plainly, no borrowed claims): LSEG News Analytics
ships proprietary novelty metadata on its own feed (noveltyCounts over
12H/24H/3D/5D/7D trailing windows, computed from linguistic fingerprints).
This module is an independent, open, MIT-licensed reimplementation on
GDELT news text using frozen FinBERT embeddings instead of linguistic
fingerprints. The empirical question tested here is the INTERACTION:
whether novelty-modulated sentiment predicts next-day abnormal returns
out-of-sample under purged walk-forward cross-validation. LSEG ships the
metadata fields; we test the predictive hypothesis openly, with numbers,
reproducible end to end.

Construction (per article i about ticker T published at time t):

- priors(i) = same-ticker articles published in [t - window_days, t),
  STRICTLY before t (an article published at exactly t is not a prior,
  and neither is anything after t).
- novelty_i = 1 - max cosine similarity between i's FinBERT embedding and
  each prior's embedding. No priors in the window -> novelty = 1.0
  (maximally novel).

New features:

- ``nov_novelty``: novelty_i as above.
- ``nov_sentiment_x_novelty``: lm_polarity (signed LM-lexicon sentiment in
  (-1, 1)) * novelty_i.
- ``nov_abs_sentiment_x_novelty``: |lm_polarity| * novelty_i (strong
  sentiment on a novel story, regardless of direction).

No-leakage rule: novelty for article i uses only articles published
strictly before i, so computing novelty once over the full frame and
splitting into walk-forward folds afterwards is safe (the same way labels
are computed once and split afterwards). ``tests/test_novelty.py`` pins
this with a toy timeline: adding a future article must not change any
earlier article's score.

Missing-embedding policy: FinBERT embeddings come from the on-disk
``.npy`` cache only (torch/transformers are optional dependencies; this
module never triggers a download or a forward pass). An article whose
embedding is not cached cannot be compared to anything, so its novelty
is 1.0 -- no measurable prior similarity means we do not discount it --
and the interaction terms fall back to unmodulated sentiment for that
row. The coverage (cached/total) is logged so the dilution is visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.models.embeddings import (
    _cache_stem,
    cache_key,
    default_cache_dir,
)
from signal_lab.models.lexicon import LEXICON_FEATURE_NAMES, lexicon_feature_matrix

NOVELTY_FEATURE_NAMES = [
    "nov_novelty",
    "nov_sentiment_x_novelty",
    "nov_abs_sentiment_x_novelty",
]

#: Trailing window (calendar days) for the prior-article set. 7D matches one
#: of the windows used by commercial novelty metadata.
NOVELTY_WINDOW_DAYS = 7


def load_embedding(text: str, cache_dir=None) -> np.ndarray | None:
    """Cached FinBERT embedding for ``text``, or None on a cache miss.

    Cache-only by design: never imports torch/transformers, never downloads
    weights, never runs a forward pass. A miss returns None and the caller
    applies the missing-embedding policy (novelty 1.0).
    """
    cdir = default_cache_dir() if cache_dir is None else cache_dir
    path = cdir / f"{_cache_stem(cache_key(text))}.npy"
    if not path.exists():
        return None
    try:
        vec = np.load(path)
    except Exception:  # noqa: BLE001 - corrupt entry behaves like a miss
        return None
    if vec.shape != (768,):
        return None
    return vec.astype(np.float32)


def compute_novelty_scores(
    tickers,
    published_at,
    embeddings,
    window_days: int = NOVELTY_WINDOW_DAYS,
) -> np.ndarray:
    """Pure novelty computation: 1 - max cosine similarity to priors.

    ``tickers`` / ``published_at``: per-article ticker and publish time.
    ``embeddings``: sequence of (768,) float arrays or None (cache miss).
    For article i, priors are same-ticker articles with publish time in
    [t_i - window_days, t_i) -- strictly before t_i, never i itself.
    Returns float64 novelty per article; NaN where the article's own
    embedding is missing (the caller maps NaN to the 1.0 policy).
    """
    tickers = np.asarray(tickers, dtype=str)
    t = pd.to_datetime(published_at, utc=True).values.astype("datetime64[ns]")
    n = len(tickers)
    if not (len(t) == n and len(embeddings) == n):
        raise ValueError("tickers, published_at and embeddings must align")
    if window_days < 1:
        raise ValueError(f"window_days must be >= 1, got {window_days}")

    window_ns = np.int64(window_days) * np.int64(86_400_000_000_000)
    t_int = t.astype(np.int64)

    out = np.full(n, np.nan, dtype=np.float64)
    for ticker in np.unique(tickers):
        m = np.where(tickers == ticker)[0]
        # Stable time order; ties keep input order but are never priors of
        # each other (the upper bound below uses side="left").
        order = m[np.argsort(t_int[m], kind="stable")]
        tt = t_int[order]
        # Normalized embedding matrix for this ticker; missing -> zero row.
        E = np.zeros((len(order), 768), dtype=np.float64)
        has = np.zeros(len(order), dtype=bool)
        for k, idx in enumerate(order):
            vec = embeddings[idx]
            if vec is not None:
                v = np.asarray(vec, dtype=np.float64)
                norm = np.linalg.norm(v)
                if norm > 0:
                    E[k] = v / norm
                    has[k] = True
        for k, idx in enumerate(order):
            if not has[k]:
                continue  # own embedding missing -> NaN (caller policy)
            lo = int(np.searchsorted(tt, tt[k] - window_ns, side="left"))
            hi = int(np.searchsorted(tt, tt[k], side="left"))
            # Candidate priors: [lo, hi), excluding k itself (k >= hi when
            # no ties; with ties k may fall inside, so mask it explicitly).
            cand = [j for j in range(lo, hi) if j != k and has[j]]
            if not cand:
                out[idx] = 1.0
                continue
            sims = E[k] @ E[cand].T
            out[idx] = float(np.clip(1.0 - np.max(sims), 0.0, 2.0))
    return out


def build_novelty_features(
    news: pd.DataFrame,
    texts=None,
    cache_dir=None,
    window_days: int = NOVELTY_WINDOW_DAYS,
    log=print,
) -> pd.DataFrame:
    """Compute the novelty feature family for each row of ``news``.

    ``news`` needs columns ``ticker`` and ``published_at``; ``texts`` is
    the article text per row (defaults to ``news["text"]``) used for both
    the embedding cache lookup and the LM-lexicon polarity.
    Returns a DataFrame with ``NOVELTY_FEATURE_NAMES`` columns on
    ``news.index``. Missing embeddings map to novelty 1.0 (documented
    policy: no measurable prior similarity, so the story is not
    discounted; the interaction terms reduce to unmodulated sentiment).
    Raises if any output is NaN -- dense features must be NaN-free before
    they reach the classifier.
    """
    if texts is None:
        texts = news["text"].fillna("").astype(str)
    texts = [str(s) for s in texts]
    if len(texts) != len(news):
        raise ValueError("texts must have one entry per news row")

    cdir = default_cache_dir() if cache_dir is None else cache_dir
    embeddings = [load_embedding(s, cache_dir=cdir) for s in texts]
    n_cached = sum(e is not None for e in embeddings)
    log(
        f"[nov] embeddings: {n_cached}/{len(news)} cached "
        f"({len(news) - n_cached} rows default to novelty 1.0)"
    )

    novelty = compute_novelty_scores(
        news["ticker"].to_numpy(), news["published_at"], embeddings,
        window_days=window_days,
    )
    # Missing-embedding policy: no measurable prior similarity -> fully
    # novel. The interaction then falls back to unmodulated sentiment.
    novelty = np.where(np.isnan(novelty), 1.0, novelty)

    polarity = lexicon_feature_matrix(texts)[
        :, LEXICON_FEATURE_NAMES.index("lm_polarity")
    ].astype(np.float64)

    frame = pd.DataFrame(
        {
            "nov_novelty": novelty,
            "nov_sentiment_x_novelty": polarity * novelty,
            "nov_abs_sentiment_x_novelty": np.abs(polarity) * novelty,
        },
        index=news.index,
    )
    if frame.isna().any().any():
        bad = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"novelty features contain NaN in columns {bad}")
    return frame[NOVELTY_FEATURE_NAMES]
