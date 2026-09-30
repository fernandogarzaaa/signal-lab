"""CLI: python -m signal_lab.nlp.run --json

Prints ONLY the JSON evaluation to stdout (logs to stderr):
{"baseline": {"precision":..,"recall":..,"f1":..},
 "ner": {...},
 "examples": [{"text":..,"entities":[..]}, ...up to 5]}

Requires data/labels.csv to be hand-labeled (see evaluate --sample).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from signal_lab.nlp import extract
from signal_lab import sanitize_json
from signal_lab.nlp.evaluate import LABELS_CSV, score, extract_baseline, extract_ner


def run_evaluation(log=print) -> dict:
    if not LABELS_CSV.exists():
        raise SystemExit(f"missing {LABELS_CSV}; label it first via "
                         f"python -m signal_lab.nlp.evaluate --sample 100")
    rows = list(csv.DictReader(open(LABELS_CSV, encoding="utf-8")))
    unlabeled = [r for r in rows if not (r["true_tickers"] or "").strip()]
    if unlabeled:
        raise SystemExit(f"{len(unlabeled)} headlines still unlabeled in {LABELS_CSV}")
    log(f"[nlp] scoring {len(rows)} labeled headlines", file=sys.stderr)

    out = {}
    for name, fn in [("baseline", extract_baseline), ("ner", extract_ner)]:
        try:
            s = score(fn)
        except RuntimeError as exc:
            # e.g. the spaCy model is not installed on this machine:
            # report honestly and keep the baseline numbers.
            log(f"[nlp] {name}: skipped ({exc})", file=sys.stderr)
            out[name] = {"precision": None, "recall": None, "f1": None,
                         "n": len(rows), "skipped": str(exc)}
            continue
        out[name] = {"precision": round(s["precision"], 4),
                     "recall": round(s["recall"], 4),
                     "f1": round(s["f1"], 4),
                     "n": s["n"]}
        log(f"[nlp] {name}: P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}",
            file=sys.stderr)

    examples = []
    for r in rows[:5]:
        hits = extract(r["title"] or "")
        examples.append({
            "text": r["title"],
            "true_tickers": r["true_tickers"],
            "entities": [{"mention": m, "ticker": t, "confidence": round(c, 2)}
                         for m, t, c in hits],
        })
    out["examples"] = examples
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate entity extractors")
    ap.add_argument("--json", action="store_true",
                    help="print only the JSON result to stdout (logs to stderr)")
    args = ap.parse_args()
    result = run_evaluation()
    if args.json:
        print(json.dumps(sanitize_json(result)))
    else:
        print(json.dumps(sanitize_json(result), indent=2))


if __name__ == "__main__":
    main()
