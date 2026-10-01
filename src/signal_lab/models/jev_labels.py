"""Label articles with TypeSafe Jev (LLM-as-annotator).

Jev is NOT a gold labeler: it cannot validate itself. The honest use is
as a scalable label source, validated against a small human sample.
Workflow:
  1. ``label`` sends each article to Jev and stores the typed choice.
  2. ``agree`` compares Jev labels against weak labels (no human needed).
  3. A human labels a small subset; Jev-vs-human kappa decides whether
     Jev labels are trustworthy enough to train on.

Usage:
    python -m signal_lab.models.jev_labels label \
        --in-path data/gold_set_150.json --out data/jev_labels_150.json
    python -m signal_lab.models.jev_labels agree \
        --in-path data/jev_labels_150.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

TYPESAFE_CLI = Path.home() / "workspace/skills/typesafe/bin/typesafe.py"

QUESTION_ID = "sentiment"
INSTRUCTIONS = (
    "Read the news headline and snippet. What is the sentiment of this news "
    "for the company named? Choose positive if it is good news for the company "
    "(earnings beat, upgrade, strong growth, positive outlook). Choose negative "
    "if it is bad news (earnings miss, downgrade, scandal, weak outlook). "
    "Choose neutral if there is no clear directional sentiment (routine "
    "announcements, mixed signals, or content not really about the company)."
)
CRITERIA = {
    "positive": "good news for the company",
    "negative": "bad news for the company",
    "neutral": "no clear directional sentiment for the company",
}
LABEL_TO_INT = {"positive": 1, "negative": 0, "neutral": 2}


def jev_decide(state: str, timeout: int = 60) -> dict:
    """Ask Jev one sentiment question; return the raw answer dict."""
    out = subprocess.run(
        [sys.executable, str(TYPESAFE_CLI), "decide",
         "--state", state,
         "--question-id", QUESTION_ID,
         "--type", "choice",
         "--instructions", INSTRUCTIONS,
         "--criteria", json.dumps(CRITERIA),
         "--timeout", str(timeout)],
        capture_output=True, text=True, timeout=timeout + 30,
    )
    if out.returncode != 0:
        raise RuntimeError(f"typesafe CLI failed: {out.stderr[:300]}")
    return json.loads(out.stdout)["answers"][QUESTION_ID]


def cmd_label(args, log=print) -> Path:
    in_path = Path(args.in_path)
    out_path = Path(args.out)
    payload = json.loads(in_path.read_text())
    items = payload["items"]
    todo = [it for it in items if it.get("jev_label") is None]
    log(f"[jev] {len(items) - len(todo)}/{len(items)} already labeled; "
        f"{len(todo)} remaining.")
    labeled = 0
    total_in, total_out = 0, 0
    for it in todo:
        state = f"Ticker: {it.get('ticker', '')}\nTitle: {it.get('title', '')}\n" \
                f"Snippet: {(it.get('body_snippet') or '')[:800]}"
        for attempt in range(3):
            try:
                ans = jev_decide(state)
                break
            except Exception as e:  # noqa: BLE001 - retry transient failures
                log(f"[jev] attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Jev failed 3x on {it.get('url')}")
        it["jev_label"] = ans["choice"]
        it["jev_label_int"] = LABEL_TO_INT[ans["choice"]]
        it["jev_confidence"] = ans["confidence"]
        it["jev_probabilities"] = ans["probabilities"]
        it["jev_model"] = payload.get("jev_model", "jev-latest")
        labeled += 1
        out_path.write_text(json.dumps(payload, indent=2))
        if labeled % 10 == 0:
            log(f"[jev] {labeled}/{len(todo)} done")
    log(f"[jev] labeled {labeled} items -> {out_path}")
    return out_path


def cmd_agree(args, log=print) -> dict:
    """Jev labels vs weak labels. No human judgment involved."""
    payload = json.loads(Path(args.in_path).read_text())
    items = payload["items"]
    judged = [it for it in items
              if it.get("jev_label_int") in (0, 1)
              and it.get("weak_label") in (0, 1)]
    n_neutral = sum(1 for it in items if it.get("jev_label_int") == 2)
    n = len(judged)
    if n == 0:
        return {"n_items": len(items), "n_judged": 0, "n_jev_neutral": n_neutral,
                "agreement": None}
    weak = [int(it["weak_label"]) for it in judged]
    jev = [int(it["jev_label_int"]) for it in judged]
    agree = sum(w == j for w, j in zip(weak, jev)) / n
    tp = sum(w == 1 and j == 1 for w, j in zip(weak, jev))
    tn = sum(w == 0 and j == 0 for w, j in zip(weak, jev))
    fp = sum(w == 1 and j == 0 for w, j in zip(weak, jev))
    fn = sum(w == 0 and j == 1 for w, j in zip(weak, jev))
    conf = [it["jev_confidence"] for it in judged]
    report = {
        "n_items": len(items),
        "n_judged": n,
        "n_jev_neutral": n_neutral,
        "agreement": round(agree, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "weak_positive_rate": round(sum(weak) / n, 4),
        "jev_positive_rate": round(sum(jev) / n, 4),
        "mean_jev_confidence": round(sum(conf) / n, 4),
    }
    log(f"[jev] Jev-vs-weak agreement: {report['agreement']} "
        f"(n={n}, neutral={n_neutral})")
    log(f"[jev] weak pos rate {report['weak_positive_rate']}, "
        f"jev pos rate {report['jev_positive_rate']}, "
        f"mean confidence {report['mean_jev_confidence']}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Label articles with TypeSafe Jev")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("label", help="send articles to Jev")
    pl.add_argument("--in-path", required=True)
    pl.add_argument("--out", required=True)
    pl.set_defaults(fn=cmd_label)

    pa = sub.add_parser("agree", help="Jev vs weak label agreement")
    pa.add_argument("--in-path", required=True)
    pa.set_defaults(fn=cmd_agree)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
