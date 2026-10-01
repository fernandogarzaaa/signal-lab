"""Training-label source selection: weak (price-derived) vs Jev-judged labels.

Weak labels are the reproducible default: free and deterministic from
market data. Jev labels are an opt-in challenger. The 2026-10-01 retrain
experiment (purged walk-forward, same folds, gold articles excluded from
training) found Jev-trained models beat weak-trained models on 4/5 folds
under both weak-label and human-gold test metrics, despite training on
far fewer rows (Jev abstains on neutrals).

Jev stays opt-in, never the default: Jev labels need a TypeSafe API key,
cost money per article, and live in git-ignored data, so a fresh clone
cannot reproduce them. The default training path must be the one anyone
can run for free.

Design rule, mirrored from the experiment: the training labels may
change, but the evaluation target never does. Test metrics are always
computed against weak labels, whatever trained the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

TRAIN_LABEL_SOURCES = ("weak", "jev", "jev-conf06")

#: Column added by :func:`apply_train_labels`. Float 0/1; NaN means Jev
#: abstained (neutral or below the confidence cutoff) and the row is
#: excluded from *training only*. Test rows always keep weak labels.
TRAIN_LABEL_COL = "train_label"

DEFAULT_JEV_LABELS_PATH = Path("data") / "jev_labels_full.json"

_JEV_TO_INT = {"positive": 1.0, "negative": 0.0}


def load_jev_labels(path: str | Path) -> pd.DataFrame:
    """Load Jev labeling output into a url-keyed frame.

    Raises FileNotFoundError with a remediation hint when the file is
    absent: silent fallback to weak labels would misreport what the
    model trained on.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Jev labels not found at {p}. Generate them with:\n"
            f"  python scripts/build_jev_full_set.py && "
            f"python -m signal_lab.models.jev_labels label "
            f"--in-path data/jev_labels_full.json "
            f"--out data/jev_labels_full.json\n"
            f"or train with --train-labels weak (the default)."
        )
    payload = json.loads(p.read_text())
    rows = [
        {
            "url": it["url"],
            "jev_label": it.get("jev_label"),
            "jev_confidence": it.get("jev_confidence"),
        }
        for it in payload["items"]
    ]
    return pd.DataFrame(rows)


def apply_train_labels(
    df: pd.DataFrame,
    source: str,
    jev_path: str | Path | None = None,
    log=print,
) -> pd.DataFrame:
    """Return ``df`` with the ``train_label`` column for ``source``.

    - ``weak``: df returned unchanged; :func:`train` uses ``label``.
    - ``jev``: directional Jev labels only; Jev neutrals become NaN
      (excluded from training, kept for testing).
    - ``jev-conf06``: as ``jev``, plus directional labels with
      confidence < 0.6 become NaN (abstention on the noisy boundary).

    Raises ValueError for an unknown source or when a dataset url has
    no Jev label: training on a partially-covered pool would silently
    shrink the training set.
    """
    if source not in TRAIN_LABEL_SOURCES:
        raise ValueError(
            f"unknown train-label source {source!r}; "
            f"expected one of {TRAIN_LABEL_SOURCES}"
        )
    if source == "weak":
        return df
    jev = load_jev_labels(jev_path or DEFAULT_JEV_LABELS_PATH)
    work = df.merge(jev, on="url", how="left")
    missing = int(work["jev_label"].isna().sum())
    if missing:
        raise ValueError(
            f"{missing}/{len(work)} dataset rows have no Jev label; "
            "label the full pool before training on Jev labels."
        )
    work[TRAIN_LABEL_COL] = work["jev_label"].map(_JEV_TO_INT)
    n_directional = int(work[TRAIN_LABEL_COL].notna().sum())
    if source == "jev-conf06":
        low_conf = work[TRAIN_LABEL_COL].notna() & (work["jev_confidence"] < 0.6)
        work.loc[low_conf, TRAIN_LABEL_COL] = float("nan")
        n_directional = int(work[TRAIN_LABEL_COL].notna().sum())
        log(
            f"[train-labels] source=jev-conf06: {n_directional} directional "
            f"rows (confidence >= 0.6), {int(low_conf.sum())} abstained on "
            "low confidence"
        )
    else:
        log(
            f"[train-labels] source=jev: {n_directional} directional rows, "
            f"{len(work) - n_directional} Jev-neutral rows excluded from "
            "training only"
        )
    if n_directional == 0:
        raise ValueError("no directional Jev labels to train on")
    pos_rate = float(work.loc[work[TRAIN_LABEL_COL].notna(), TRAIN_LABEL_COL].mean())
    log(f"[train-labels] Jev training positive rate: {pos_rate:.3f}")
    return work.drop(columns=["jev_label", "jev_confidence"])
