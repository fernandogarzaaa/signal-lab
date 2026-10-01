"""Tests for the interactive gold-set label command (0.4.0)."""

import json
from argparse import Namespace
from pathlib import Path

from signal_lab.models.gold_set import GOLD_VERSION, cmd_label


def _payload(tmp_path: Path, n: int = 3) -> Path:
    items = [
        {
            "url": f"https://x.test/{i}",
            "title": f"Headline {i}",
            "body_snippet": f"Snippet body {i} " * 10,
            "published_at": "2026-05-0%dT12:00:00+00:00" % (i + 1),
            "ticker": "AAPL",
            "weak_label": i % 2,
            "human_label": None,
            "human_note": "",
        }
        for i in range(n)
    ]
    p = tmp_path / "gold.json"
    p.write_text(json.dumps({"version": GOLD_VERSION, "items": items}))
    return p


def test_label_marks_items_and_saves(tmp_path, monkeypatch, capsys):
    p = _payload(tmp_path, 2)
    answers = iter(["1 h", "0 m beat"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    out = cmd_label(Namespace(in_path=str(p)), log=lambda *a, **k: None)
    assert out["labeled_now"] == 2
    assert out["labeled_total"] == 2
    data = json.loads(p.read_text())
    assert [it["human_label"] for it in data["items"]] == [1, 0]
    assert data["items"][1]["human_note"] == "beat"


def test_label_never_shows_weak_label(tmp_path, monkeypatch, capsys):
    """Blinding: the weak label must not appear in what the human sees."""
    p = _payload(tmp_path, 1)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "q")
    out_lines: list[str] = []
    cmd_label(Namespace(in_path=str(p)), log=out_lines.append)
    shown = "\n".join(out_lines)
    # weak labels are 0/1 but must not be presented as the answer key;
    # specifically the JSON field name must never leak into the UI.
    assert "weak_label" not in shown


def test_label_quit_keeps_progress(tmp_path, monkeypatch):
    p = _payload(tmp_path, 3)
    answers = iter(["1 h", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    out = cmd_label(Namespace(in_path=str(p)), log=lambda *a, **k: None)
    assert out["labeled_now"] == 1
    data = json.loads(p.read_text())
    assert data["items"][0]["human_label"] == 1
    assert data["items"][1]["human_label"] is None


def test_label_skip_leaves_null(tmp_path, monkeypatch):
    p = _payload(tmp_path, 2)
    answers = iter(["s", "0 h"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    cmd_label(Namespace(in_path=str(p)), log=lambda *a, **k: None)
    data = json.loads(p.read_text())
    assert data["items"][0]["human_label"] is None
    assert data["items"][1]["human_label"] == 0


def test_label_skips_already_labeled(tmp_path, monkeypatch):
    p = _payload(tmp_path, 2)
    data = json.loads(p.read_text())
    data["items"][0]["human_label"] = 1
    p.write_text(json.dumps(data))
    seen: list[str] = []
    monkeypatch.setattr("builtins.input", lambda *a, **k: seen.append("x") or "0 h")
    out = cmd_label(Namespace(in_path=str(p)), log=lambda *a, **k: None)
    assert out["labeled_now"] == 1  # only the unlabeled item was presented
    assert len(seen) == 1


def test_label_neutral_with_confidence(tmp_path, monkeypatch):
    p = _payload(tmp_path, 2)
    answers = iter(["2 m mixed signals", "1 h"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    cmd_label(Namespace(in_path=str(p)), log=lambda *a, **k: None)
    data = json.loads(p.read_text())
    assert data["items"][0]["human_label"] == 2
    assert data["items"][0]["human_confidence"] == "m"
    assert data["items"][0]["human_note"] == "mixed signals"
    assert data["items"][1]["human_confidence"] == "h"


def test_label_permanent_skip(tmp_path, monkeypatch):
    p = _payload(tmp_path, 2)
    answers = iter(["x", "1 h"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    out = cmd_label(Namespace(in_path=str(p)), log=lambda *a, **k: None)
    data = json.loads(p.read_text())
    assert data["items"][0]["human_label"] == "x"
    assert out["labeled_now"] == 1
