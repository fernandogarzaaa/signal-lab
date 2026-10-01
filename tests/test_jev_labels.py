"""Tests for the Jev LLM-as-annotator labeling (no API calls)."""

import json
from argparse import Namespace
from pathlib import Path

from signal_lab.models.jev_labels import cmd_agree


def _payload(tmp_path: Path) -> Path:
    items = [
        {"url": "https://x.test/0", "weak_label": 1,
         "jev_label": "positive", "jev_label_int": 1, "jev_confidence": 0.9,
         "jev_probabilities": {"positive": 0.9, "negative": 0.05,
                               "neutral": 0.05}},
        {"url": "https://x.test/1", "weak_label": 0,
         "jev_label": "positive", "jev_label_int": 1, "jev_confidence": 0.8,
         "jev_probabilities": {"positive": 0.8, "negative": 0.1,
                               "neutral": 0.1}},
        {"url": "https://x.test/2", "weak_label": 0,
         "jev_label": "neutral", "jev_label_int": 2, "jev_confidence": 0.7,
         "jev_probabilities": {"positive": 0.2, "negative": 0.2,
                               "neutral": 0.6}},
        {"url": "https://x.test/3", "weak_label": 0,
         "jev_label": None, "jev_label_int": None, "jev_confidence": None,
         "jev_probabilities": None},
    ]
    p = tmp_path / "jev.json"
    p.write_text(json.dumps({"items": items}))
    return p


def test_agree_excludes_neutral_and_unlabeled(tmp_path):
    p = _payload(tmp_path)
    out = cmd_agree(Namespace(in_path=str(p)), log=lambda *a, **k: None)
    assert out["n_items"] == 4
    assert out["n_judged"] == 2  # only binary Jev labels with weak labels
    assert out["n_jev_neutral"] == 1
    assert out["agreement"] == 0.5  # 1 of 2 agree
    assert out["tp"] == 1
    assert out["fn"] == 1  # weak=0, jev=1 (weak missed a Jev positive)
    assert out["weak_positive_rate"] == 0.5
    assert out["jev_positive_rate"] == 1.0
