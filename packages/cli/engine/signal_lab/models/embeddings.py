"""0.3.0 workstream 2: FinBERT text representation (optional dependency).

ProsusAI/finbert is used as a FROZEN feature extractor: each article maps
to one 768-dim vector (CLS pooling by default, attention-masked mean
pooling on request). The vectors feed the same logistic-regression
machinery as the TF-IDF + lexicon features; see ``run_finbert_ablation``
for the side-by-side comparison (TF-IDF-only vs FinBERT-only vs combined,
same data, same temporal split).

Optional-dependency design
--------------------------
torch + transformers are NOT hard requirements. Everything here degrades
cleanly when they are missing (or when the ~440MB model has never been
downloaded): ``finbert_available()`` returns False, ``embed_texts()``
raises ``FinbertUnavailableError``, and ``run_finbert_ablation()`` logs a
clear line and reports only the TF-IDF arm. The seeded demo, ``doctor``,
and the default stage-3 run all work fully offline.

First use downloads ~440MB of weights from the Hugging Face hub
(``ProsusAI/finbert``) into the standard HF cache; per-article embeddings
are then cached as ``.npy`` files under ``~/.cache/signal_lab/finbert``
(override with ``SIGNAL_LAB_FINBERT_CACHE``) keyed by SHA-256 of the
normalized text (or an explicit key such as the article URL), so re-runs
never recompute.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

FINBERT_MODEL_ID = "ProsusAI/finbert"
EMBEDDING_DIM = 768
FINBERT_ENABLE_ENV = "SIGNAL_LAB_FINBERT"  # "1" also opts stage 3 into the ablation
FINBERT_CACHE_ENV = "SIGNAL_LAB_FINBERT_CACHE"
RANDOM_STATE = 7

_WS_RE = re.compile(r"\s+")
_SAFE_STEM = re.compile(r"[A-Za-z0-9_.\-]{1,128}\Z")


def _cache_stem(key: str) -> str:
    """Filename stem for a cache key: used verbatim when filesystem-safe,
    otherwise the SHA-256 of the key (URLs fall in the second bucket)."""
    if _SAFE_STEM.match(key):
        return key
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class FinbertUnavailableError(RuntimeError):
    """Raised when FinBERT embeddings are requested but unavailable."""


def _import_torch():
    try:
        import torch
    except Exception as exc:
        raise FinbertUnavailableError(
            "torch is not installed; install it (e.g. `pip install torch`) "
            "to enable FinBERT embeddings"
        ) from exc
    import torch

    return torch


def _import_transformers():
    try:
        import transformers
    except Exception as exc:
        raise FinbertUnavailableError(
            "transformers is not installed; install it "
            "(e.g. `pip install transformers`) to enable FinBERT embeddings"
        ) from exc
    import transformers

    return transformers


def _deps_installed() -> tuple[bool, bool]:
    """(torch_installed, transformers_installed) without raising."""
    try:
        import torch  # noqa: F401

        torch_ok = True
    except Exception:  # noqa: BLE001
        torch_ok = False
    try:
        import transformers  # noqa: F401

        tf_ok = True
    except Exception:  # noqa: BLE001
        tf_ok = False
    return torch_ok, tf_ok


def _hf_model_dir(model_id: str = FINBERT_MODEL_ID) -> Path | None:
    """Local HF-hub cache dir for the model, or None if never downloaded.

    Probes the filesystem only: no network, no transformers import.
    """
    cache_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not cache_home:
        cache_home = str(Path.home() / ".cache" / "huggingface" / "hub")
    repo_dir = Path(cache_home) / f"models--{model_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    for snap in sorted(snapshots.iterdir()):
        if (snap / "config.json").exists():
            return snap
    return None


def model_cached(model_id: str = FINBERT_MODEL_ID) -> bool:
    """True when the model weights are already on disk (offline-usable)."""
    return _hf_model_dir(model_id) is not None


def finbert_available() -> bool:
    """True when embeddings can be produced right now without any download
    (deps installed AND weights already cached)."""
    torch_ok, tf_ok = _deps_installed()
    return bool(torch_ok and tf_ok and model_cached())


def deps_installed() -> bool:
    """True when torch + transformers import (weights may still download)."""
    torch_ok, tf_ok = _deps_installed()
    return bool(torch_ok and tf_ok)


def availability_report() -> dict:
    """JSON-safe dict for the ``doctor`` FinBERT check."""
    torch_ok, tf_ok = _deps_installed()
    cached = model_cached()
    available = bool(torch_ok and tf_ok and cached)
    if available:
        detail = f"{FINBERT_MODEL_ID} ready (deps installed, weights cached)"
    elif not torch_ok:
        detail = "torch not installed"
    elif not tf_ok:
        detail = "transformers not installed"
    else:
        detail = f"{FINBERT_MODEL_ID} weights not downloaded yet (~440MB on first use)"
    return {
        "model": FINBERT_MODEL_ID,
        "embedding_dim": EMBEDDING_DIM,
        "torch_installed": torch_ok,
        "transformers_installed": tf_ok,
        "model_cached": cached,
        "available": available,
        "detail": detail,
    }


def default_cache_dir() -> Path:
    """Where per-article embedding .npy files live."""
    override = os.environ.get(FINBERT_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "signal_lab" / "finbert"


def cache_key(text: str) -> str:
    """Stable cache key: SHA-256 of whitespace-normalized text."""
    normalized = _WS_RE.sub(" ", str(text).strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class FinbertExtractor:
    """Lazy, frozen ProsusAI/finbert feature extractor with a disk cache.

    The model (tokenizer + weights) loads on first ``embed()`` call, runs in
    eval mode under ``torch.no_grad()`` (frozen: no fine-tuning, per the
    0.3.0 non-goals), and each article's vector is cached as
    ``{key}.npy`` under the cache dir so re-runs never recompute.
    """

    def __init__(
        self,
        model_id: str = FINBERT_MODEL_ID,
        pooling: str = "cls",
        device: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        if pooling not in ("cls", "mean"):
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.model_id = model_id
        self.pooling = pooling
        self.device = device
        self.cache_dir = (
            Path(cache_dir).expanduser()
            if cache_dir is not None
            else default_cache_dir()
        )
        self._tokenizer = None
        self._model = None
        self._torch_device = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        torch = _import_torch()
        transformers = _import_transformers()
        if _hf_model_dir(self.model_id) is None:
            log.warning(
                "[finbert] downloading %s weights (~440MB) from the "
                "Hugging Face hub; subsequent runs use the local cache",
                self.model_id,
            )
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_device = torch.device(device)
        self._torch = torch
        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            self._model = transformers.AutoModel.from_pretrained(self.model_id)
        except Exception as exc:
            raise FinbertUnavailableError(
                f"could not load {self.model_id} ({exc}); "
                "check the network connection and retry, or stay on the "
                "TF-IDF+lexicon fallback"
            ) from exc
        self._model.to(self._torch_device)
        self._model.eval()
        log.info(
            "FinBERT loaded: %s on %s (frozen, %s pooling)",
            self.model_id,
            device,
            self.pooling,
        )

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """Raw (n, 768) float32 embeddings for texts already known to be
        cache misses. Separated so the cache layer is unit-testable without
        torch."""
        if self._torch is None or self._model is None:
            raise FinbertUnavailableError(
                "extractor not loaded; call embed() instead of _encode_batch()"
            )
        torch = self._torch

        enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(self._torch_device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc).last_hidden_state  # (n, seq, 768)
        if self.pooling == "cls":
            pooled = out[:, 0, :]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled.detach().cpu().numpy().astype(np.float32)

    def _read_cached(self, key: str) -> np.ndarray | None:
        path = self.cache_dir / f"{_cache_stem(key)}.npy"
        if not path.exists():
            return None
        try:
            vec = np.load(path)
        except Exception:  # noqa: BLE001 - corrupt cache entry -> recompute
            return None
        if vec.shape != (EMBEDDING_DIM,):
            return None
        return vec.astype(np.float32)

    def embed(
        self,
        texts,
        keys=None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """(n, 768) float32 matrix, input order preserved.

        ``keys`` optionally overrides the cache key per text (e.g. article
        URLs); otherwise the key is the SHA-256 of the normalized text.
        """
        texts = [str(t) for t in texts]
        if keys is None:
            keys = [cache_key(t) for t in texts]
        else:
            keys = [str(k) for k in keys]
        if len(keys) != len(texts):
            raise ValueError("keys and texts must have the same length")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        out: list[np.ndarray | None] = [None] * len(texts)
        misses: list[int] = []
        for i, key in enumerate(keys):
            hit = self._read_cached(key)
            if hit is not None:
                out[i] = hit
            else:
                misses.append(i)
        if misses:
            self._ensure_loaded()
            for start in range(0, len(misses), batch_size):
                chunk = misses[start : start + batch_size]
                vecs = self._encode_batch([texts[i] for i in chunk])
                for i, vec in zip(chunk, vecs):
                    (self.cache_dir / f"{_cache_stem(keys[i])}.npy").write_bytes(
                        _npy_bytes(vec)
                    )
                    out[i] = vec
        return np.stack([np.asarray(v, dtype=np.float32) for v in out])

    def cache_stats(self) -> dict:
        """How many embeddings are cached and how much disk they use."""
        if not self.cache_dir.is_dir():
            return {"entries": 0, "bytes": 0, "cache_dir": str(self.cache_dir)}
        files = list(self.cache_dir.glob("*.npy"))
        return {
            "entries": len(files),
            "bytes": sum(f.stat().st_size for f in files),
            "cache_dir": str(self.cache_dir),
        }


def _npy_bytes(vec: np.ndarray) -> bytes:
    """Serialize one vector to .npy bytes without a second dependency."""
    import io

    buf = io.BytesIO()
    np.save(buf, np.asarray(vec, dtype=np.float32))
    return buf.getvalue()


def embed_texts(
    texts,
    keys=None,
    cache_dir: str | Path | None = None,
    log=print,
) -> np.ndarray:
    """Embed texts with FinBERT, or raise FinbertUnavailableError.

    Missing weights download on first use (~440MB, logged loudly); only a
    missing torch/transformers install or a failed download raises. The
    caller decides the fallback (TF-IDF+lexicon); this keeps the "heavy
    dependency missing" failure mode explicit at every call site.
    """
    if not deps_installed():
        raise FinbertUnavailableError(availability_report()["detail"])
    extractor = FinbertExtractor(cache_dir=cache_dir)
    vecs = extractor.embed(texts, keys=keys)
    stats = extractor.cache_stats()
    log(
        f"[finbert] embedded {len(vecs)} articles "
        f"({stats['entries']} cached entries, "
        f"{stats['bytes'] / 1e6:.1f} MB in {stats['cache_dir']})"
    )
    return vecs


def finbert_feature_matrix(
    texts,
    keys=None,
    cache_dir: str | Path | None = None,
    log=print,
) -> np.ndarray:
    """Dense (n, 768) FinBERT feature matrix (raises if unavailable)."""
    return embed_texts(texts, keys=keys, cache_dir=cache_dir, log=log)


def combined_feature_matrix(
    texts,
    vectorizer,
    keys=None,
    cache_dir: str | Path | None = None,
    log=print,
):
    """Sparse TF-IDF + 8 LM lexicon + 768 FinBERT columns (raises if the
    FinBERT side is unavailable)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse import hstack as sparse_hstack

    from signal_lab.models import featurize

    base = featurize(texts, vectorizer)
    emb = finbert_feature_matrix(texts, keys=keys, cache_dir=cache_dir, log=log)
    return sparse_hstack([base, csr_matrix(emb)], format="csr")


def _arm_metrics(name: str, y_true: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y_pred = (scores >= 0.5).astype(int)
    return {
        "name": name,
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "threshold": 0.5,
    }


def _fit_logreg(Xtr, ytr):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    clf.fit(Xtr, ytr)
    return clf


def run_finbert_ablation(
    df,
    text_col: str = "text",
    log=print,
    tfidf_result: dict | None = None,
    cache_dir: str | Path | None = None,
) -> dict:
    """Ablation: TF-IDF+lexicon vs FinBERT-only vs combined.

    Same dataframe, same temporal train/test split, same classifier family
    (class-weighted logistic regression, threshold 0.5) in every arm, so
    differences measure the representation, not the protocol. When FinBERT
    is unavailable the finbert/combined arms are skipped with a clear log
    line and the TF-IDF arm still reports.
    """
    import pandas as pd
    from scipy.sparse import csr_matrix
    from scipy.sparse import hstack as sparse_hstack
    from sklearn.preprocessing import StandardScaler

    from signal_lab.models import featurize, make_text, temporal_split, train

    df = df.dropna(subset=[text_col, "label", "published_at"]).reset_index(drop=True)
    if text_col == "title":
        texts = make_text(df)
    else:
        texts = df[text_col].fillna("").astype(str)
    y = df["label"].astype(int).values
    tr_idx, te_idx = temporal_split(pd.to_datetime(df["published_at"]))
    n_train, n_test = len(tr_idx), len(te_idx)

    arms: dict[str, dict] = {}

    # Arm 1: TF-IDF + LM lexicon (the 0.2.0 representation). Reuse the
    # pipeline's own train() result when handed in to avoid training twice.
    if tfidf_result is None:
        tfidf_result = train(df, text_col=text_col)
    by_name = {r.name: r for r in tfidf_result["reports"]}
    rep = by_name["logreg_balanced"]
    arms["tfidf_lexicon"] = {
        "name": "tfidf_lexicon",
        "precision": round(rep.precision, 4),
        "recall": round(rep.recall, 4),
        "f1": round(rep.f1, 4),
        "pr_auc": round(rep.pr_auc, 4),
        "accuracy": round(rep.accuracy, 4),
        "threshold": rep.threshold,
    }
    vectorizer = tfidf_result["vectorizer"]

    meta = {
        "model": FINBERT_MODEL_ID,
        "embedding_dim": EMBEDDING_DIM,
        "pooling": "cls",
        "n_train": n_train,
        "n_test": n_test,
        "positive_rate": round(float(y.mean()), 4),
        "arms": arms,
    }

    if not deps_installed():
        reason = availability_report()["detail"]
        log(
            "[m3] FinBERT unavailable "
            f"({reason}); finbert/combined arms skipped, "
            "falling back to TF-IDF+lexicon"
        )
        meta["available"] = False
        meta["reason"] = reason
        return meta

    stats_line = ""
    try:
        extractor = FinbertExtractor(cache_dir=cache_dir)
        E = extractor.embed(list(texts))
        stats = extractor.cache_stats()
        stats_line = f"{stats['entries']} cached entries, {stats['bytes'] / 1e6:.1f} MB"
    except FinbertUnavailableError as exc:
        log(f"[m3] FinBERT embedding failed ({exc}); arms skipped")
        meta["available"] = False
        meta["reason"] = str(exc)
        return meta

    log(f"[m3] FinBERT embeddings ready: {E.shape}, {stats_line}")

    # Arm 2: FinBERT-only (dense; standardized for the linear model).
    scaler = StandardScaler().fit(E[tr_idx])
    Xtr_f = scaler.transform(E[tr_idx])
    Xte_f = scaler.transform(E[te_idx])
    clf_f = _fit_logreg(Xtr_f, y[tr_idx])
    arms["finbert_only"] = _arm_metrics(
        "finbert_only", y[te_idx], clf_f.predict_proba(Xte_f)[:, 1]
    )

    # Arm 3: combined TF-IDF + lexicon + FinBERT.
    Xtr_c = sparse_hstack(
        [featurize(texts.iloc[tr_idx], vectorizer), csr_matrix(Xtr_f)],
        format="csr",
    )
    Xte_c = sparse_hstack(
        [featurize(texts.iloc[te_idx], vectorizer), csr_matrix(Xte_f)],
        format="csr",
    )
    clf_c = _fit_logreg(Xtr_c, y[tr_idx])
    arms["combined"] = _arm_metrics(
        "combined", y[te_idx], clf_c.predict_proba(Xte_c)[:, 1]
    )

    meta["available"] = True
    meta["cache_dir"] = str(extractor.cache_dir)
    log(
        "[m3] finbert ablation "
        + " | ".join(
            f"{name}: PR-AUC {a['pr_auc']:.4f}, F1 {a['f1']:.4f}"
            for name, a in arms.items()
        )
    )
    return meta


def ablation_table(ablation: dict) -> str:
    """One-line-per-arm text table for logs and the model card."""
    lines = ["arm | PR-AUC | F1 | precision | recall | n_train | n_test"]
    arms = ablation.get("arms", {})
    n_train = ablation.get("n_train", "?")
    n_test = ablation.get("n_test", "?")
    for name, a in arms.items():
        lines.append(
            f"{name} | {a['pr_auc']:.4f} | {a['f1']:.4f} | "
            f"{a['precision']:.4f} | {a['recall']:.4f} | {n_train} | {n_test}"
        )
    return "\n".join(lines)


def save_ablation_json(ablation: dict, path: str | Path) -> Path:
    """Persist the ablation block (JSON-safe) for the model card."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ablation, indent=2, default=str))
    return path


def finbert_ctx_matrix(
    texts,
    dense_extra=None,
    cache_dir: str | Path | None = None,
    log=print,
) -> "np.ndarray | csr_matrix":
    """FinBERT (768-dim, cache-only) + dense context features.

    0.4.0: the walk-forward winner. Cache-only by design: never imports
    torch/transformers, never downloads. Texts without a cached embedding
    become zero vectors (count logged). ``dense_extra`` (e.g. the 7 market-
    context features) is hstacked when provided, giving a sparse matrix;
    without it the result is the dense (n, 768) embedding matrix.
    """
    import numpy as np

    from signal_lab.models.novelty import load_embedding

    vecs = []
    missing = 0
    for t in texts:
        v = load_embedding(str(t), cache_dir=cache_dir)
        if v is None:
            missing += 1
            v = np.zeros(768, dtype=np.float32)
        vecs.append(v)
    emb = np.stack(vecs)
    log(f"[finbert] cache-only matrix: {emb.shape[0]} rows, "
        f"{missing} zero-filled (no torch import, no download)")
    if dense_extra is None:
        return emb
    from scipy.sparse import csr_matrix
    from scipy.sparse import hstack as sparse_hstack

    return sparse_hstack(
        [csr_matrix(emb), csr_matrix(np.asarray(dense_extra, dtype=float))],
        format="csr",
    )
