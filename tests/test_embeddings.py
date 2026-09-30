"""0.3.0 workstream 2: FinBERT embeddings, cache, fallback, ablation.

Offline-safe by default: every test except ``test_embedding_shape_real``
runs without torch/transformers or the model weights. The real-embedding
test skips unless FinBERT is actually available.
"""

import sys

import numpy as np
import pandas as pd
import pytest

from signal_lab.models import embeddings
from signal_lab.models.embeddings import (
    EMBEDDING_DIM,
    FINBERT_MODEL_ID,
    FinbertExtractor,
    FinbertUnavailableError,
    availability_report,
    cache_key,
    embed_texts,
    finbert_available,
    run_finbert_ablation,
)


@pytest.fixture
def no_torch(monkeypatch):
    """Simulate an environment without the optional heavy dependencies."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    yield


def _fixture(n=120):
    rows = []
    for i in range(n):
        good = i % 3 == 0
        rows.append(
            {
                "title": "record earnings beat, strong revenue growth"
                if good
                else "company announces routine board meeting",
                "body_snippet": "",
                "label": 1 if good else 0,
                "published_at": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            }
        )
    return pd.DataFrame(rows)


# --- availability / fallback -------------------------------------------------


def test_availability_report_shape(no_torch):
    rep = availability_report()
    assert rep["model"] == FINBERT_MODEL_ID
    assert rep["embedding_dim"] == EMBEDDING_DIM
    assert rep["torch_installed"] is False
    assert rep["transformers_installed"] is False
    assert rep["available"] is False
    assert isinstance(rep["detail"], str) and rep["detail"]


def test_finbert_available_false_without_deps(no_torch):
    assert finbert_available() is False


def test_embed_texts_raises_when_unavailable(no_torch):
    with pytest.raises(FinbertUnavailableError):
        embed_texts(["some finance text"])


def test_cache_key_deterministic():
    assert cache_key("  hello   world ") == cache_key("hello world")
    assert cache_key("aaa") != cache_key("aab")
    assert len(cache_key("x")) == 64


def test_extractor_rejects_bad_pooling(tmp_path):
    with pytest.raises(ValueError):
        FinbertExtractor(pooling="max", cache_dir=tmp_path)


# --- cache hit/miss (no torch: override the encode step) ----------------------


class _FakeEncoder(FinbertExtractor):
    """Extractor with a deterministic stand-in for the torch forward pass."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encode_calls = 0

    def _ensure_loaded(self):
        return  # no torch needed for the fake encoder

    def _encode_batch(self, texts):
        self.encode_calls += 1
        rng = np.random.RandomState(abs(hash(texts[0])) % (2**31))
        return rng.rand(len(texts), EMBEDDING_DIM).astype(np.float32)


def test_cache_miss_then_hit(tmp_path):
    ext = _FakeEncoder(cache_dir=tmp_path)
    texts = ["earnings beat raises guidance", "board meeting adjourned"]
    first = ext.embed(texts)
    assert first.shape == (2, EMBEDDING_DIM)
    assert first.dtype == np.float32
    assert ext.encode_calls == 1
    assert (tmp_path / f"{cache_key(texts[0])}.npy").exists()

    # Fresh extractor, same cache dir: everything must be a hit.
    ext2 = _FakeEncoder(cache_dir=tmp_path)
    second = ext2.embed(texts)
    assert ext2.encode_calls == 0
    np.testing.assert_array_equal(first, second)
    stats = ext2.cache_stats()
    assert stats["entries"] == 2 and stats["bytes"] > 0


def test_cache_keys_override(tmp_path):
    import hashlib

    ext = _FakeEncoder(cache_dir=tmp_path)
    ext.embed(["same text", "same text"], keys=["http://x/1", "http://x/2"])
    # URL keys are hashed to filesystem-safe stems, one entry per key
    for url in ("http://x/1", "http://x/2"):
        stem = hashlib.sha256(url.encode()).hexdigest()
        assert (tmp_path / f"{stem}.npy").exists()
    assert ext.cache_stats()["entries"] == 2


def test_cache_key_length_mismatch(tmp_path):
    ext = _FakeEncoder(cache_dir=tmp_path)
    with pytest.raises(ValueError):
        ext.embed(["a", "b"], keys=["only-one"])


def test_corrupt_cache_entry_recomputed(tmp_path):
    ext = _FakeEncoder(cache_dir=tmp_path)
    (tmp_path / f"{cache_key('hello')}.npy").write_bytes(b"not a numpy file")
    vecs = ext.embed(["hello"])
    assert vecs.shape == (1, EMBEDDING_DIM)
    assert ext.encode_calls == 1  # corrupt entry treated as a miss


def test_wrong_shape_cache_entry_recomputed(tmp_path):
    ext = _FakeEncoder(cache_dir=tmp_path)
    np.save(tmp_path / f"{cache_key('hello')}.npy", np.zeros(16))
    vecs = ext.embed(["hello"])
    assert vecs.shape == (1, EMBEDDING_DIM)
    assert ext.encode_calls == 1


# --- ablation runner ----------------------------------------------------------


def test_ablation_offline_fallback(no_torch):
    logs = []
    df = _fixture()
    out = run_finbert_ablation(df, text_col="title", log=logs.append)
    assert out["available"] is False
    assert out.get("reason")
    assert set(out["arms"]) == {"tfidf_lexicon"}
    arm = out["arms"]["tfidf_lexicon"]
    for k in ("precision", "recall", "f1", "pr_auc", "accuracy"):
        assert 0.0 <= arm[k] <= 1.0
    assert out["n_train"] > 0 and out["n_test"] > 0
    assert any("FinBERT unavailable" in line for line in logs), logs


class _FakeAvailableExtractor:
    """Stand-in wired through the real ablation path (no torch)."""

    cache_dir = "/tmp/fake-finbert-cache"

    def __init__(self, cache_dir=None):
        if cache_dir is not None:
            self.cache_dir = str(cache_dir)

    def embed(self, texts, keys=None, batch_size=32):
        rng = np.random.RandomState(1234)
        return rng.rand(len(texts), EMBEDDING_DIM).astype(np.float32)

    def cache_stats(self):
        return {"entries": 0, "bytes": 0, "cache_dir": self.cache_dir}


def test_ablation_three_arms_with_fake_embeddings(monkeypatch, tmp_path):
    monkeypatch.setattr(embeddings, "deps_installed", lambda: True)
    monkeypatch.setattr(embeddings, "FinbertExtractor", _FakeAvailableExtractor)
    logs = []
    df = _fixture()
    out = run_finbert_ablation(
        df, text_col="title", log=logs.append, cache_dir=tmp_path
    )
    assert out["available"] is True
    assert out["model"] == FINBERT_MODEL_ID
    assert set(out["arms"]) == {"tfidf_lexicon", "finbert_only", "combined"}
    for name, arm in out["arms"].items():
        for k in ("precision", "recall", "f1", "pr_auc", "accuracy"):
            assert 0.0 <= arm[k] <= 1.0, (name, k, arm[k])
    # same split in every arm
    assert out["n_train"] == 84 and out["n_test"] == 36
    assert any("finbert ablation" in line for line in logs), logs


def test_maybe_run_ablation_not_requested():
    from signal_lab.models.build_and_train import maybe_run_finbert_ablation

    out = maybe_run_finbert_ablation(_fixture(), requested=False)
    assert out["available"] is False
    assert "not requested" in out["reason"]


def test_maybe_run_ablation_requested_offline(no_torch):
    from signal_lab.models.build_and_train import maybe_run_finbert_ablation

    logs = []
    df = _fixture()
    df["text"] = df["title"]  # run_pipeline always supplies a text column
    out = maybe_run_finbert_ablation(df, log=logs.append, requested=True)
    assert out["available"] is False
    assert set(out["arms"]) == {"tfidf_lexicon"}


# --- registry + predict wiring -------------------------------------------------


def test_feature_builders_registry():
    from signal_lab.models.build_and_train import FEATURE_BUILDERS

    assert set(FEATURE_BUILDERS) >= {
        "tfidf_lexicon",
        "finbert_only",
        "tfidf_lexicon_finbert",
    }
    with pytest.raises(ValueError):
        from signal_lab.models.build_and_train import register_feature_builder

        register_feature_builder("tfidf_lexicon", lambda t, v: None)


def test_predict_with_feature_builder():
    from signal_lab.models import featurize, predict, train
    from signal_lab.models.build_and_train import FEATURE_BUILDERS

    df = _fixture()
    result = train(df, text_col="title")
    model = result["models"]["logreg_balanced"]
    vec = result["vectorizer"]
    default = predict(model, ["record earnings beat"], vectorizer=vec)
    via_registry = predict(
        model,
        ["record earnings beat"],
        vectorizer=vec,
        feature_builder=FEATURE_BUILDERS["tfidf_lexicon"],
    )
    assert default == via_registry
    # a custom builder is honored too
    via_custom = predict(
        model,
        ["record earnings beat"],
        vectorizer=vec,
        feature_builder=lambda texts, v: featurize(texts, v),
    )
    assert via_custom == default


def test_ablation_table_and_save(tmp_path, no_torch):
    from signal_lab.models.embeddings import ablation_table, save_ablation_json

    logs = []
    out = run_finbert_ablation(_fixture(), text_col="title", log=logs.append)
    table = ablation_table(out)
    assert "tfidf_lexicon" in table and "PR-AUC" in table
    assert "0.1395" not in table  # fixture numbers, not the real run's
    path = save_ablation_json(out, tmp_path / "abl.json")
    assert path.exists()
    import json

    assert json.loads(path.read_text())["arms"]["tfidf_lexicon"]["f1"] >= 0.0


# --- real embedding (only when the model is actually available) ----------------


@pytest.mark.skipif(
    not finbert_available(), reason="FinBERT weights/deps not available"
)
def test_embedding_shape_real():
    vecs = embed_texts(
        [
            "Apple reports record quarterly earnings",
            "The board announced a routine meeting",
        ],
        log=lambda *a, **k: None,
    )
    assert vecs.shape == (2, EMBEDDING_DIM)
    assert vecs.dtype == np.float32
    assert np.isfinite(vecs).all()
    # deterministic: same text -> same vector
    again = embed_texts(
        ["Apple reports record quarterly earnings"], log=lambda *a, **k: None
    )
    np.testing.assert_allclose(vecs[0], again[0], rtol=1e-5, atol=1e-6)
