"""Loughran-McDonald finance sentiment lexicon features.

The Loughran-McDonald Master Dictionary (2025) is the standard word list
for financial text sentiment (Loughran & McDonald, 2011, Journal of Finance).
General-purpose lexicons misfire on finance text ("liability", "tax", "cost"
are not negative in 10-Ks); LM was built from 10-K filings and encodes that
domain knowledge. Bundled as package data (models/data/lm_lexicon.json),
extracted from the authoritative Master Dictionary CSV.

Per article we emit 8 dense features:
  lm_pos_count, lm_neg_count, lm_unc_count,
  lm_pos_rate, lm_neg_rate, lm_unc_rate  (per token),
  lm_polarity       = (pos - neg) / (pos + neg + 1),
  lm_subjectivity   = (pos + neg) / n_tokens.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
# Legacy location (local dev runs); package data is preferred.
LEXICON_JSON = REPO_ROOT / "data" / "lm_lexicon.json"

LEXICON_FEATURE_NAMES = [
    "lm_pos_count", "lm_neg_count", "lm_unc_count",
    "lm_pos_rate", "lm_neg_rate", "lm_unc_rate",
    "lm_polarity", "lm_subjectivity",
]

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _load_lexicon() -> dict[str, set[str]]:
    path = LEXICON_JSON
    if not path.exists():
        # Bundled package data (works installed, in CI, in the npm engine).
        path = resources.files(__package__) / "data" / "lm_lexicon.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: set(v) for k, v in raw.items()}


@lru_cache(maxsize=1)
def lexicon() -> dict[str, set[str]]:
    """Word sets keyed by category: positive, negative, uncertainty, ..."""
    return _load_lexicon()


def lexicon_feature_matrix(texts) -> np.ndarray:
    """(n_texts, 8) float32 feature matrix for an iterable of strings."""
    lex = lexicon()
    pos_w, neg_w, unc_w = lex["positive"], lex["negative"], lex["uncertainty"]
    rows = []
    for text in texts:
        toks = _WORD_RE.findall(str(text).lower())
        n = len(toks)
        pc = sum(1 for t in toks if t in pos_w)
        nc = sum(1 for t in toks if t in neg_w)
        uc = sum(1 for t in toks if t in unc_w)
        rows.append([
            float(pc), float(nc), float(uc),
            pc / n if n else 0.0,
            nc / n if n else 0.0,
            uc / n if n else 0.0,
            (pc - nc) / (pc + nc + 1),
            (pc + nc) / n if n else 0.0,
        ])
    return np.asarray(rows, dtype=np.float32)
