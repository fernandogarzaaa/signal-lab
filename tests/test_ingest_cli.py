"""Regression tests for the ingest CLI (no network, no DB).

Covers the --json output path of ``python -m signal_lab.ingest.run``,
which crashed with NameError: sanitize_json (missing import) in the
0.2.0 release.
"""

import io
import json
from contextlib import redirect_stdout
from unittest import mock

import pytest


def _run_main_json(monkeypatch, summary):
    from signal_lab.ingest import run as run_mod

    monkeypatch.setattr(
        run_mod, "run_ingest", lambda *a, **k: summary
    )
    monkeypatch.setattr("sys.argv", ["run.py", "--tickers", "AAPL", "--json"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_mod.main()
    return buf.getvalue()


def test_main_json_prints_valid_json(monkeypatch):
    summary = {
        "news_rows": 0,
        "price_rows": 2,
        "tickers": ["AAPL"],
        "start": "2026-09-29",
        "end": "2026-09-30",
        "news_windows": 0,
    }
    out = _run_main_json(monkeypatch, summary)
    parsed = json.loads(out)
    assert parsed["price_rows"] == 2
    assert parsed["tickers"] == ["AAPL"]


def test_main_json_sanitizes_nan_to_null(monkeypatch):
    summary = {
        "news_rows": 0,
        "price_rows": 1,
        "tickers": ["AAPL"],
        "start": "2026-09-29",
        "end": "2026-09-30",
        "news_windows": 0,
        "avg_sentiment": float("nan"),
    }
    out = _run_main_json(monkeypatch, summary)
    parsed = json.loads(out)
    assert parsed["avg_sentiment"] is None


def test_sanitize_json_importable_from_ingest_run():
    from signal_lab.ingest import run as run_mod

    assert callable(run_mod.sanitize_json)
