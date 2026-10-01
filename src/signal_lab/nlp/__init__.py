"""M2: entity extraction. Company mention -> ticker (PermID-lite).

Two extractors:
  extract_baseline: deterministic alias-dictionary matching (data/aliases.csv).
  extract_ner:      spaCy NER (ORG entities) mapped to tickers via aliases,
                   with difflib fuzzy fallback.

Ambiguity rule: single common-word aliases ("apple", "meta", "amazon") only
count at full confidence when a finance-context word (stock, shares,
earnings, revenue, market, investor, CEO, ...) appears nearby; otherwise
they are kept at low confidence. Multi-word aliases are unambiguous.
"""

from __future__ import annotations

import csv
import difflib
import re
from functools import lru_cache
from importlib import resources
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
# Legacy location (local dev runs); package data is preferred.
ALIASES_CSV = REPO_ROOT / "data" / "aliases.csv"

# Words that mark a sentence as finance/business context.
FINANCE_WORDS = {
    "stock", "stocks", "share", "shares", "earnings", "revenue", "revenues",
    "profit", "loss", "market", "markets", "investor", "investors", "ceo",
    "nasdaq", "nyse", "dow", "s&p", "ipo", "dividend", "analyst", "wall street",
    "trading", "trade", "billion", "million", "quarter", "guidance", "forecast",
    "chip", "chips", "ai",
}

# Single-word aliases that are ambiguous outside finance context.
AMBIGUOUS = {
    "apple", "meta", "amazon", "alphabet", "tesla", "nvidia", "exxon",
    # Gate-2 universe additions (single common words; same finance-context rule).
    "visa", "booking", "cat", "now", "ba", "usb", "hd", "pm",
}


def _load_aliases() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    # Prefer the CSV bundled as package data (works installed, in CI, and in
    # the npm-packaged engine). Fall back to the repo-local data/ copy.
    csv_path = ALIASES_CSV
    if not csv_path.exists():
        csv_path = resources.files(__package__) / "data" / "aliases.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            alias = row["alias"].strip().lower()
            ticker = row["ticker"].strip().upper()
            if alias and ticker:
                rows.append((alias, ticker))
    # longest alias first so "apple inc" wins over "apple"
    rows.sort(key=lambda r: -len(r[0]))
    return rows


@lru_cache(maxsize=1)
def aliases() -> list[tuple[str, str]]:
    return _load_aliases()


def alias_map() -> dict[str, str]:
    return {a: t for a, t in aliases()}


def _finance_context(text: str, start: int, end: int, window: int = 120) -> bool:
    ctx = text[max(0, start - window): end + window].lower()
    return any(w in ctx for w in FINANCE_WORDS)


def extract_baseline(text: str) -> list[tuple[str, str, float]]:
    """Deterministic alias matching. Returns (mention, ticker, confidence)."""
    out: list[tuple[str, str, float]] = []
    claimed: list[tuple[int, int]] = []
    for alias, ticker in aliases():
        for m in re.finditer(r"\b" + re.escape(alias) + r"\b", text, re.IGNORECASE):
            s, e = m.span()
            if any(s < ce and e > cs for cs, ce in claimed):
                continue  # overlapped by a longer alias already claimed
            claimed.append((s, e))
            if alias in AMBIGUOUS and not _finance_context(text, s, e):
                conf = 0.5
            else:
                conf = 0.95
            out.append((m.group(0), ticker, conf))
    return out


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed; run "
            "'python -m spacy download en_core_web_sm'. "
            "extract_baseline() works without it."
        ) from exc


def _normalize_org(name: str) -> str:
    name = name.lower()
    name = re.sub(r"\b(inc|corp|corporation|ltd|limited|co|company|group|holdings)\b\.?", "", name)
    return " ".join(name.split())


def extract_ner(text: str) -> list[tuple[str, str, float]]:
    """spaCy entities mapped to tickers via the alias dictionary.

    Labels beyond ORG (GPE, PRODUCT, FAC) are included because the small
    spaCy model mislabels companies ("Nvidia" -> GPE). The alias dictionary
    is the whitelist: only mentions that resolve to a known ticker are
    returned, so wider labels cannot invent tickers.
    """
    amap = alias_map()
    out: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for ent in _nlp()(text).ents:
        if ent.label_ not in {"ORG", "GPE", "PRODUCT", "FAC"}:
            continue
        norm = _normalize_org(ent.text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        if norm in amap:
            out.append((ent.text, amap[norm], 0.9))
            continue
        close = difflib.get_close_matches(norm, amap.keys(), n=1, cutoff=0.82)
        if close:
            out.append((ent.text, amap[close[0]], 0.7))
    return out


def extract(text: str) -> list[tuple[str, str, float]]:
    """Default extractor: NER, falling back to baseline when NER finds nothing
    or the spaCy model is unavailable."""
    try:
        hits = extract_ner(text)
    except RuntimeError:
        hits = []
    return hits if hits else extract_baseline(text)
