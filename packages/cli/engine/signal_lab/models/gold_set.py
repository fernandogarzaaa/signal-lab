"""CLI: python -m signal_lab.models.gold_set sample|agree

Gold-set tooling for 0.3.0 workstream 1 (label quality). The weak labels are
noisy; the gold set quantifies that noise with human judgment.

Workflow:
  1. ``sample`` draws N articles (stratified by weak label) and writes a
     JSON file with one item per article.
  2. A human (Inan) opens the file and fills in ``human_label`` per item:
     1 = the article reads positive for the company, 0 = negative,
     null = skip. ``human_note`` is optional free text.
  3. ``agree`` reads the file back and reports the weak-label agreement
     rate, a confusion breakdown, and Cohen's kappa.

Storage format (version 1):
  {"version": 1, "created": "<utc iso>", "label_config": {...},
   "weak_label_source": "<human-readable description>",
   "instructions": "<how to fill human_label>",
   "items": [{"url", "title", "body_snippet", "published_at", "ticker",
              "weak_label": 0|1, "human_label": 0|1|null,
              "human_note": ""}, ...]}

Example:
  python -m signal_lab.models.gold_set sample --n 150 --seed 7 \\
      --out data/gold_set.json
  # ... human labels the file ...
  python -m signal_lab.models.gold_set agree --in data/gold_set.json
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from signal_lab import sanitize_json
from signal_lab.models.run import add_label_args, label_config_from_args

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLD_VERSION = 1
INSTRUCTIONS = (
    "For each item, read the headline and snippet and answer: does this read "
    "POSITIVE (1) or NEGATIVE (0) for the company named in 'ticker'? Set "
    "human_label to 1 or 0. Leave it null to skip an item you cannot judge. "
    "'human_note' is optional free text explaining a judgment call."
)


@contextmanager
def _db_override(path: str | None):
    """Point stage-3 DB access at another DuckDB file for this block."""
    if path is None:
        yield
        return
    import signal_lab.models.build_and_train as bat
    from signal_lab import ingest

    old_ingest, old_bat = ingest.DB_PATH, bat.DB_PATH
    ingest.DB_PATH = bat.DB_PATH = path
    try:
        yield
    finally:
        ingest.DB_PATH, bat.DB_PATH = old_ingest, old_bat


def sample_gold_set(
    df: pd.DataFrame, n: int, seed: int, pos_frac: float = 0.5
) -> pd.DataFrame:
    """Draw n rows stratified by weak label (pos_frac positives), shuffled
    deterministically by seed. df needs columns url and label."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if not 0.0 <= pos_frac <= 1.0:
        raise ValueError("pos_frac must be in [0, 1]")
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    n_pos = min(len(pos), round(n * pos_frac))
    n_neg = min(len(neg), n - n_pos)
    n_pos = min(len(pos), n - n_neg)  # refill if the negative class ran short
    take = pd.concat(
        [
            pos.sample(n=n_pos, random_state=seed) if n_pos else pos.iloc[0:0],
            neg.sample(n=n_neg, random_state=seed + 1) if n_neg else neg.iloc[0:0],
        ]
    )
    return take.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


def cohens_kappa(weak: list[int], human: list[int]) -> float:
    """Cohen's kappa between weak labels and human labels."""
    n = len(weak)
    po = sum(w == h for w, h in zip(weak, human)) / n
    pw1 = sum(weak) / n
    ph1 = sum(human) / n
    pe = pw1 * ph1 + (1.0 - pw1) * (1.0 - ph1)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def agreement_report(items: list[dict]) -> dict:
    """Agreement between weak labels and human labels.

    Each item needs 'weak_label' and 'human_label'. Items whose human_label
    is not 0/1 (null = skipped) are excluded. The human label is treated as
    ground truth: tp/tn/fp/fn describe the weak label against it.
    """
    judged = [it for it in items if it.get("human_label") in (0, 1)]
    n = len(judged)
    if n == 0:
        return {
            "n_items": len(items),
            "n_judged": 0,
            "agreement": None,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "cohens_kappa": None,
            "weak_positive_rate": None,
            "human_positive_rate": None,
        }
    weak = [int(it["weak_label"]) for it in judged]
    human = [int(it["human_label"]) for it in judged]
    tp = sum(w == 1 and h == 1 for w, h in zip(weak, human))
    tn = sum(w == 0 and h == 0 for w, h in zip(weak, human))
    fp = sum(w == 1 and h == 0 for w, h in zip(weak, human))
    fn = sum(w == 0 and h == 1 for w, h in zip(weak, human))
    return {
        "n_items": len(items),
        "n_judged": n,
        "agreement": round((tp + tn) / n, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "cohens_kappa": round(cohens_kappa(weak, human), 4),
        "weak_positive_rate": round(sum(weak) / n, 4),
        "human_positive_rate": round(sum(human) / n, 4),
    }


def _display_fields(db_path: str, urls: list[str]) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute(
        "SELECT url, title, body_snippet, published_at FROM news_raw "
        "WHERE url IN (SELECT unnest(?))",
        [urls],
    ).fetchdf()
    con.close()
    return rows.set_index("url")


def cmd_sample(args, log=print) -> Path:
    from signal_lab.models.build_and_train import build_dataset

    label_cfg = label_config_from_args(args)
    with _db_override(args.db):
        from signal_lab import ingest

        df, label_block = build_dataset(log=log, label_cfg=label_cfg)
        db_path = ingest.DB_PATH
    log(f"[gold] labeled pool: {len(df)} rows")
    sampled = sample_gold_set(df, args.n, args.seed, pos_frac=args.pos_frac)
    display = _display_fields(db_path, sampled["url"].tolist())
    items = []
    for _, row in sampled.iterrows():
        if row["url"] in display.index:
            disp = display.loc[row["url"]]
            title = str(disp.get("title", "") or "")
            snippet = str(disp.get("body_snippet", "") or "")
        else:
            title, snippet = "", ""
        pub = row["published_at"]
        items.append(
            {
                "url": row["url"],
                "title": title,
                "body_snippet": snippet,
                "published_at": pub.isoformat()
                if hasattr(pub, "isoformat")
                else str(pub),
                "ticker": row["ticker"],
                "weak_label": int(row["label"]),
                "human_label": None,
                "human_note": "",
            }
        )
    payload = {
        "version": GOLD_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "label_config": label_block,
        "weak_label_source": (
            f"{label_block['scheme']}, {label_block['window_days']} trading "
            f"day(s), top {label_block['quantile']:.0%} abnormal return vs "
            f"{label_block['benchmark']}"
        ),
        "instructions": INSTRUCTIONS,
        "items": items,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sanitize_json(payload), indent=2))
    log(f"[gold] wrote {len(items)} items -> {out}")
    return out


def cmd_agree(args, log=print) -> dict:
    payload = json.loads(Path(args.in_path).read_text())
    if payload.get("version") != GOLD_VERSION:
        raise ValueError(
            f"unsupported gold-set version {payload.get('version')!r}; "
            f"expected {GOLD_VERSION}"
        )
    report = agreement_report(payload["items"])
    report["label_config"] = payload.get("label_config")
    if args.json:
        print(json.dumps(sanitize_json(report)))
    else:
        if report["n_judged"] == 0:
            log(
                "[gold] no human labels yet: fill in 'human_label' (0/1) "
                "in the gold-set file, then re-run agree."
            )
        else:
            log("gold-set agreement: weak labels vs human labels")
            log(f"  judged: {report['n_judged']}/{report['n_items']} items")
            log(f"  agreement rate: {report['agreement']:.3f}")
            log(
                f"  confusion (weak vs human): "
                f"tp={report['tp']} tn={report['tn']} "
                f"fp={report['fp']} fn={report['fn']}"
            )
            log(f"  cohen's kappa: {report['cohens_kappa']:.3f}")
            log(
                f"  positive rate: weak={report['weak_positive_rate']:.3f} "
                f"human={report['human_positive_rate']:.3f}"
            )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Gold-set sampling and agreement")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("sample", help="sample articles for hand-labeling")
    sp.add_argument("--n", type=int, default=150)
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument(
        "--pos-frac",
        type=float,
        default=0.5,
        help="fraction of the sample drawn from weak positives",
    )
    sp.add_argument("--out", default=str(REPO_ROOT / "data" / "gold_set.json"))
    sp.add_argument(
        "--db", default=None, help="DuckDB path override (default: the stage-3 DB)"
    )
    add_label_args(sp)

    ag = sub.add_parser("agree", help="report weak-vs-human agreement")
    ag.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="gold-set JSON file with human labels filled in",
    )
    ag.add_argument(
        "--json", action="store_true", help="print only the JSON report to stdout"
    )

    args = ap.parse_args()
    log = (
        (lambda *a, **k: print(*a, file=sys.stderr, **k))
        if getattr(args, "json", False)
        else print
    )
    if args.cmd == "sample":
        cmd_sample(args, log=log)
    else:
        cmd_agree(args, log=log)


if __name__ == "__main__":
    main()
