"""0.4.0 experiment: FinBERT embeddings in purged walk-forward.

Compares text representations under the honest 3-fold purged walk-forward:
  (a) TF-IDF + lexicon + 7 ctx        (current baseline)
  (e) FinBERT-only (768-dim, cache-only; zeros on cache miss)
  (f) FinBERT + 7 ctx
  (g) TF-IDF + lexicon + FinBERT + 7 ctx

Cache-only by design: never imports torch/transformers. Missing embeddings
become zero vectors (logged). Embeddings are a deterministic function of
text, so precomputing them before the fold split leaks no label information.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sparse_hstack

sys.path.insert(0, "/home/hatch/workspace/signal-lab/src")

from signal_lab.models import featurize, make_text
from signal_lab.models.build_and_train import (
    build_dataset,
    load_trading_calendar_days,
)
from signal_lab.models.novelty import load_embedding
from signal_lab.models.run import add_label_args, label_config_from_args
from signal_lab.models.walk_forward import run_walk_forward

ARTIFACT = Path("/home/hatch/workspace/signal-lab/data/artifacts/"
                "walk_forward_finbert.json")


def build_embedding_lookup(texts: pd.Series):
    lookup = {}
    missing = 0
    for t in texts.unique():
        vec = load_embedding(str(t))
        if vec is None:
            missing += 1
            vec = np.zeros(768, dtype=np.float32)
        lookup[str(t)] = vec
    print(f"[finbert-exp] cache coverage: {len(lookup) - missing}/{len(lookup)} "
          f"texts ({missing} zero-filled)", flush=True)
    return lookup


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    add_label_args(ap)
    # Baseline scheme: the windowed scheme lost the benchmark (0.088 vs 0.141).
    args = ap.parse_args(["--label-scheme", "baseline"])
    label_cfg = label_config_from_args(args)

    df, label_block = build_dataset(log=print, label_cfg=label_cfg)
    print(f"[finbert-exp] labeled rows: {len(df)}", flush=True)
    df = df.reset_index(drop=True)
    # walk_forward uses df["text"] when present; key embeddings on the same.
    texts = df["text"].fillna("").astype(str) if "text" in df.columns \
        else make_text(df).fillna("").astype(str)
    lookup = build_embedding_lookup(texts)
    emb_all = np.stack([lookup[str(t)] for t in texts])

    # dense context features, if present
    ctx_cols = [c for c in df.columns if c.startswith("ctx_")]
    dense_all = df[ctx_cols].to_numpy(dtype=float) if ctx_cols else None
    print(f"[finbert-exp] ctx columns: {ctx_cols}", flush=True)

    # map from positional iloc to embedding rows: walk_forward resets index,
    # so positional alignment holds if we pass embeddings via a closure over
    # the row order. We instead key by text inside featurize_fn.
    def finbert_matrix(sub_texts):
        return np.stack([lookup[str(t)] for t in sub_texts])

    def arm_e(sub_texts, vectorizer, dense_extra=None):
        return csr_matrix(finbert_matrix(sub_texts))

    def arm_f(sub_texts, vectorizer, dense_extra=None):
        m = csr_matrix(finbert_matrix(sub_texts))
        if dense_extra is not None:
            m = sparse_hstack([m, csr_matrix(dense_extra)], format="csr")
        return m

    def arm_g(sub_texts, vectorizer, dense_extra=None):
        base = featurize(sub_texts, vectorizer, dense_extra)
        return sparse_hstack([base, csr_matrix(finbert_matrix(sub_texts))],
                             format="csr")

    arms = {
        "(a) tfidf+lexicon+ctx": None,
        "(e) finbert-only": arm_e,
        "(f) finbert+ctx": arm_f,
        "(g) tfidf+lexicon+finbert+ctx": arm_g,
    }
    results = {}
    trading_days = load_trading_calendar_days()
    for name, fn in arms.items():
        print(f"[finbert-exp] arm {name}", flush=True)
        res = run_walk_forward(
            df, dense_extra=dense_all, n_splits=3,
            trading_days=trading_days, label_cfg=label_cfg,
            embargo_days=3, featurize_fn=fn, log=lambda *a, **k: None,
        )
        pr = [f["pr_auc"] for f in res["folds"]]
        f1 = [f["f1"] for f in res["folds"]]
        results[name] = {
            "pr_auc_per_fold": [round(x, 4) for x in pr],
            "f1_per_fold": [round(x, 4) for x in f1],
            "mean_pr_auc": round(float(np.mean(pr)), 4),
            "mean_f1": round(float(np.mean(f1)), 4),
            "n_scored": len(pr),
        }
        print(f"[finbert-exp] {name}: mean PR-AUC={np.mean(pr):.4f} "
              f"folds={[round(x, 4) for x in pr]}", flush=True)

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(results, indent=2))
    print(f"[finbert-exp] wrote {ARTIFACT}", flush=True)


if __name__ == "__main__":
    main()
