"""M6: RAG explainer. Cited explanations over 10-K filings.

Corpus: latest 10-K per ticker from SEC EDGAR (free, no key). Retrieval is
TF-IDF cosine nearest-neighbor over ~800-char chunks: a deliberate,
documented choice. Dense embeddings would help paraphrase matching, but
10-K language is keyword-dense ("revenue", "risk", "impairment"), where
TF-IDF is strong, and it keeps the whole pipeline dependency-light and
fully reproducible.

Explanations are extractive: the most relevant sentences from retrieved
chunks, each cited by chunk_id. No LLM generation means no unsupported
claims by construction; every sentence traces to a retrieved chunk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[3]
EDGAR_DIR = REPO_ROOT / "data" / "edgar"
ART = REPO_ROOT / "data" / "artifacts"

UA = {"User-Agent": "SignalLab/0.1 (interview-prep research project; signal-lab-research@example.com)",
      "Accept-Encoding": "gzip, deflate"}

# CIKs for the ingest universe.
CIKS = {
    "AAPL": "0000320193", "MSFT": "0000789019", "NVDA": "0001045810",
    "TSLA": "0001318605", "AMZN": "0001018724", "GOOGL": "0001652044",
    "META": "0001326801", "JPM": "0000019617", "XOM": "0000034088",
    "JNJ": "0000200406",
}


def _get(url: str, timeout: float = 60.0) -> requests.Response:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r


def fetch_10k(ticker: str) -> str:
    """Download the latest 10-K main document text for ticker. Cached on disk."""
    cik = CIKS[ticker]
    cache = EDGAR_DIR / f"{ticker}_10k.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    subs = _get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    recent = subs["filings"]["recent"]
    acc = None
    for form, a in zip(recent["form"], recent["accessionNumber"]):
        if form == "10-K":
            acc = a.replace("-", "")
            break
    if acc is None:
        raise RuntimeError(f"no 10-K found for {ticker}")
    index = _get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/index.json").json()
    items = [i for i in index["directory"]["item"]
             if i["name"].endswith((".htm", ".html")) and "index" not in i["name"].lower()]
    if not items:
        raise RuntimeError(f"no 10-K document found for {ticker}")
    # The main 10-K is by far the largest HTML document in the filing.
    doc = max(items, key=lambda i: int(i.get("size", "0") or 0))["name"]
    html = _get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}").text
    # Drop the SEC submission header (tags are stripped later, so cut it in raw HTML).
    if "</SEC-HEADER>" in html:
        html = html.split("</SEC-HEADER>", 1)[1]
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    text = _mdna_section(text) or text[:120000]
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


def _mdna_section(text: str) -> str:
    """Extract Item 7 (MD&A) .. Item 7A/8 boundary; '' if not found.

    Uses the LAST 'Item 7' heading: the first occurrence is usually the
    table of contents, the real section comes later in the document.
    """
    starts = list(re.finditer(r"ITEM\s+7\.\s+MANAGEMENT", text, re.IGNORECASE))
    if not starts:
        return ""
    m7 = starts[-1]
    m8 = re.search(r"ITEM\s+7A\.|ITEM\s+8\.", text[m7.end():], re.IGNORECASE)
    end = m7.end() + m8.start() if m8 else m7.end() + 120000
    section = text[m7.start():end]
    return section if len(section) > 2000 else ""


def chunk_text(text: str, ticker: str, size: int = 800, overlap: int = 100) -> list[dict]:
    chunks = []
    i, n = 0, 0
    while i < len(text):
        piece = text[i:i + size]
        chunks.append({"chunk_id": f"{ticker}#{n:04d}", "ticker": ticker, "text": piece})
        n += 1
        i += size - overlap
    return chunks


def build_index(chunks: list[dict]):
    vec = TfidfVectorizer(max_features=20000, stop_words="english", sublinear_tf=True)
    mat = vec.fit_transform([c["text"] for c in chunks])
    return vec, mat


def save_index(chunks: list[dict], vec, mat) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump({"chunks": chunks, "vectorizer": vec, "matrix": mat}, ART / "rag_index.pkl")


def load_index():
    return joblib.load(ART / "rag_index.pkl")


def retrieve(query: str, k: int = 5, ticker: str | None = None) -> list[dict]:
    """Top-k chunks by TF-IDF cosine similarity, optionally ticker-filtered."""
    idx = load_index()
    chunks, vec, mat = idx["chunks"], idx["vectorizer"], idx["matrix"]
    q = vec.transform([query])
    sims = cosine_similarity(q, mat)[0]
    order = np.argsort(sims)[::-1]
    out = []
    for i in order:
        c = chunks[i]
        if ticker and c["ticker"] != ticker:
            continue
        out.append({**c, "score": float(sims[i])})
        if len(out) == k:
            break
    return out


def _key_sentences(text: str, query_terms: set[str], n: int = 2) -> list[str]:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 40]
    scored = sorted(sents, key=lambda s: -sum(t in s.lower() for t in query_terms))
    return scored[:n]


def explain(event: dict, k: int = 5) -> dict:
    """Extractive cited explanation for an event.

    event: {'ticker': str, 'event_date': str/date, 'headline': str (optional)}.
    Returns {'text': str, 'citations': [{'chunk_id', 'quote'}]}.
    """
    ticker = event["ticker"]
    headline = event.get("headline", "")
    query = f"{ticker} {headline} revenue risk earnings growth"
    terms = set(re.findall(r"[a-z]{3,}", query.lower()))
    chunks = retrieve(query, k=k, ticker=ticker)
    lines, citations = [], []
    for c in chunks:
        for s in _key_sentences(c["text"], terms):
            lines.append(f"{s} [{c['chunk_id']}]")
            citations.append({"chunk_id": c["chunk_id"], "quote": s[:220]})
            if len(lines) >= 6:
                break
        if len(lines) >= 6:
            break
    header = (f"On {event.get('event_date')}, {ticker} showed a sentiment spike"
              + (f' around: "{headline[:120]}".' if headline else ".")
              + " The company's own 10-K describes the following relevant context:")
    return {"text": header + "\n" + "\n".join(f"- {l}" for l in lines),
            "citations": citations}
