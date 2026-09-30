"""Evaluate entity extractors against hand labels.

Usage:
    python -m signal_lab.nlp.evaluate --sample 100 --seed 7   # build sample to label
    python -m signal_lab.nlp.evaluate --eval                   # score vs data/labels.csv

labels.csv format: url,title,true_tickers  (true_tickers like "AAPL;MSFT" or "NONE")
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import duckdb
import pandas as pd

from signal_lab.ingest import DB_PATH
from signal_lab.nlp import extract_baseline, extract_ner

REPO_ROOT = Path(__file__).resolve().parents[3]
LABELS_CSV = REPO_ROOT / "data" / "labels.csv"


def build_sample(n: int, seed: int) -> None:
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(
        "SELECT url, title, query FROM news_raw WHERE title <> '' ORDER BY RANDOM()"
    ).fetchdf()
    con.close()
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    sample["true_tickers"] = ""
    sample[["url", "title", "true_tickers"]].to_csv(LABELS_CSV, index=False)
    print(f"wrote {len(sample)} headlines to {LABELS_CSV} for hand labeling")


def _gold(true_tickers: str) -> set[str]:
    t = (true_tickers or "").strip().upper()
    return set() if t in ("", "NONE") else {x.strip() for x in t.split(";") if x.strip()}


def score(extractor) -> dict:
    rows = list(csv.DictReader(open(LABELS_CSV, encoding="utf-8")))
    tp = fp = fn = 0
    for r in rows:
        gold = _gold(r["true_tickers"])
        pred = {t for _, t, _ in extractor(r["title"] or "")}
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    if args.sample:
        build_sample(args.sample, args.seed)
    if args.eval:
        if not LABELS_CSV.exists():
            raise SystemExit(f"missing {LABELS_CSV}; run --sample first and label it")
        unlabeled = sum(1 for r in csv.DictReader(open(LABELS_CSV, encoding="utf-8"))
                             if not (r["true_tickers"] or "").strip())
        if unlabeled:
            raise SystemExit(f"{unlabeled} headlines still unlabeled")
        for name, fn in [("baseline", extract_baseline), ("ner", extract_ner)]:
            s = score(fn)
            print(f"{name:9s} n={s['n']} tp={s['tp']} fp={s['fp']} fn={s['fn']} "
                  f"P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}")


if __name__ == "__main__":
    main()
