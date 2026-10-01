"""Unit tests for entity extractors (no network, no DB)."""

import pytest

from signal_lab.nlp import extract_baseline, extract_ner


def test_baseline_finds_alias():
    hits = extract_baseline("Apple reported record iPhone revenue this quarter")
    tickers = {t for _, t, _ in hits}
    assert "AAPL" in tickers


def test_baseline_longest_alias_wins():
    hits = extract_baseline("Apple Inc beat expectations")
    mentions = [m for m, _, _ in hits]
    assert any("Apple Inc" in m for m in mentions)
    assert not any(m.strip().lower() == "apple" for m in mentions)


def test_ambiguous_alias_low_confidence_without_context():
    hits = extract_baseline("I ate an apple for lunch")
    apple = [h for h in hits if h[1] == "AAPL"]
    assert apple and all(c < 0.9 for _, _, c in apple)


def test_ner_extracts_org():
    pytest.importorskip("spacy", reason="spaCy is an optional dependency")
    hits = extract_ner("Microsoft and Nvidia announced a chip partnership")
    tickers = {t for _, t, _ in hits}
    assert {"MSFT", "NVDA"} <= tickers


def test_no_false_positive_on_empty():
    assert extract_baseline("") == []


def test_gate2_ambiguous_aliases_gated_by_context():
    # "visa"/"booking"/"cat" only reach full confidence with finance words nearby.
    for word, ticker in [("visa", "V"), ("booking", "BKNG"), ("cat", "CAT")]:
        low = [h for h in extract_baseline(f"my {word} expired yesterday") if h[1] == ticker]
        assert low and all(c < 0.9 for _, _, c in low), word
        high = [h for h in extract_baseline(f"{word} stock surged on earnings") if h[1] == ticker]
        assert high and all(c >= 0.9 for _, _, c in high), word


def test_gate2_universe_tickers_resolve():
    hits = extract_baseline("Berkshire Hathaway and AT&T reported earnings; Citigroup shares fell")
    tickers = {t for _, t, _ in hits}
    assert {"BRK-B", "T", "C"} <= tickers
