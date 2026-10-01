"""Tests for finbert_ctx_matrix (0.4.0): FinBERT + context, cache-only."""

import numpy as np
import pytest

from signal_lab.models import embeddings


def _fake_embedding(text: str):
    # Deterministic fake: hash the text into a 768-dim vector.
    rng = np.random.default_rng(abs(hash(text)) % (2**31))
    return rng.standard_normal(768).astype(np.float32)


def test_finbert_ctx_matrix_shapes(monkeypatch):
    monkeypatch.setattr(
        "signal_lab.models.novelty.load_embedding",
        lambda text, cache_dir=None: _fake_embedding(text),
    )
    texts = ["apple earnings beat", "tesla miss"]
    dense = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = embeddings.finbert_ctx_matrix(texts, dense_extra=dense, log=lambda *a, **k: None)
    assert out.shape == (2, 770)  # 768 + 2 ctx
    # ctx columns land at the end, unscaled
    assert out[:, -2:].toarray().tolist() == dense.tolist()


def test_finbert_ctx_matrix_zero_fills_missing_only_when_explicit(monkeypatch):
    """Zero-fill is an explicitly-named fallback, not the default."""
    monkeypatch.setattr(
        "signal_lab.models.novelty.load_embedding",
        lambda text, cache_dir=None: None,
    )
    out = embeddings.finbert_ctx_matrix(
        ["nope"], dense_extra=None, log=lambda *a, **k: None, on_missing="zero"
    )
    assert out.shape == (1, 768)
    assert (out == 0).all()


def test_finbert_ctx_matrix_raises_on_missing_by_default(monkeypatch):
    """Default policy is loud failure: MissingEmbeddingError naming the
    count and the cache directory."""
    monkeypatch.setattr(
        "signal_lab.models.novelty.load_embedding",
        lambda text, cache_dir=None: None,
    )
    with pytest.raises(embeddings.MissingEmbeddingError) as ei:
        embeddings.finbert_ctx_matrix(["nope"], log=lambda *a, **k: None)
    msg = str(ei.value)
    assert "1/1" in msg
    assert "finbert" in msg  # cache dir named
    assert "on_missing='zero'" in msg  # explicit fallback named


def test_finbert_ctx_matrix_rejects_bad_policy():
    with pytest.raises(ValueError, match="on_missing"):
        embeddings.finbert_ctx_matrix(["x"], log=lambda *a, **k: None,
                                      on_missing="bogus")


def test_finbert_ctx_matrix_no_torch_import(monkeypatch):
    """Cache-only contract: must not import torch or transformers."""
    import sys

    monkeypatch.setattr(
        "signal_lab.models.novelty.load_embedding",
        lambda text, cache_dir=None: _fake_embedding(text),
    )
    for mod in ("torch", "transformers"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    # block re-import
    import builtins

    real_import = builtins.__import__

    def guarded(name, *a, **k):
        if name.split(".")[0] in ("torch", "transformers"):
            raise ImportError(f"blocked: {name}")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guarded)
    out = embeddings.finbert_ctx_matrix(["x"], log=lambda *a, **k: None)
    assert out.shape == (1, 768)
