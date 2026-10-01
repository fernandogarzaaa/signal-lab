"""Tests for the Jev LLM-as-annotator labeling (no API calls)."""

import json
from argparse import Namespace
from pathlib import Path

from signal_lab.models.jev_labels import (
    CRITERIA,
    INSTRUCTIONS,
    LABEL_TO_INT,
    PROMPT_VERSION,
    build_state,
    cmd_agree,
)


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


def test_build_state_formats_fields():
    item = {"ticker": "AAPL", "title": "Apple beats",
            "body_snippet": "x" * 2000}
    state = build_state(item)
    assert "Ticker: AAPL" in state
    assert "Title: Apple beats" in state
    assert state.count("x") == 800  # snippet truncated


def test_build_state_handles_missing_fields():
    state = build_state({})
    assert "Ticker: " in state
    assert "Title: " in state
    assert "Snippet: " in state


def test_prompt_version_is_stamped():
    assert PROMPT_VERSION.startswith("v2-")


def test_instructions_cover_neutral_rules():
    # Regression guard: the v2 prompt must keep the neutral no-signal rules
    # learned from the 150-article human-labeling session.
    lowered = INSTRUCTIONS.lower()
    for phrase in [
        "would a shareholder",  # ticker-specific buy/sell framing
        "no hindsight",  # publication-date judgment
        "price recaps",  # backward-looking price chatter -> neutral
        "fund",  # routine fund filings -> neutral
        "different company",  # ticker mismatches -> neutral
        "choose neutral",  # tie-break toward neutral
    ]:
        assert phrase in lowered, f"missing from INSTRUCTIONS: {phrase!r}"


def test_criteria_cover_all_labels():
    assert set(CRITERIA) == set(LABEL_TO_INT) == {"positive", "negative", "neutral"}
    assert all(CRITERIA[k].strip() for k in CRITERIA)
