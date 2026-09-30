"""0.4.0: fill missing FinBERT embeddings for the 902-row labeled set.

Uses FinbertExtractor (downloads ProsusAI/finbert ~440MB on first use).
Only embeds texts not already in the cache; idempotent.
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/signal-lab/src")

from signal_lab.models.build_and_train import build_dataset
from signal_lab.models.embeddings import FinbertExtractor, cache_key, default_cache_dir
from signal_lab.models.run import add_label_args, label_config_from_args
import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    add_label_args(ap)
    args = ap.parse_args(["--label-scheme", "baseline"])
    df, _ = build_dataset(log=lambda *a, **k: None,
                          label_cfg=label_config_from_args(args))
    texts = df["text"].fillna("").astype(str).tolist()
    print(f"[fill] {len(texts)} labeled texts", flush=True)

    cdir = default_cache_dir()
    missing = []
    seen_keys = set()
    for t in texts:
        k = cache_key(t)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        if not (cdir / f"{k}.npy").exists():
            # check filesystem-safe stem variant too
            from signal_lab.models.embeddings import _cache_stem
            if not (cdir / f"{_cache_stem(k)}.npy").exists():
                missing.append(t)
    print(f"[fill] {len(missing)} unique texts need embeddings", flush=True)
    if not missing:
        print("[fill] nothing to do")
        return
    ext = FinbertExtractor()
    vecs = ext.embed(missing)
    print(f"[fill] embedded {len(vecs)}; cache now has "
          f"{len(list(cdir.glob('*.npy')))} entries", flush=True)


if __name__ == "__main__":
    main()
