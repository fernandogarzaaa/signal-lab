"""Build the M6 corpus: fetch latest 10-K per ticker, chunk, index, persist.

Usage: python -m signal_lab.rag.build_corpus [--tickers AAPL,MSFT]
"""

from __future__ import annotations

import argparse
import time

from signal_lab.rag import (CIKS, build_index, chunk_text, fetch_10k,
                            save_index)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(CIKS))
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    all_chunks = []
    for i, t in enumerate(tickers):
        if i:
            time.sleep(1)  # SEC asks for <=10 req/s; stay far below
        try:
            text = fetch_10k(t)
        except Exception as exc:
            print(f"[m6] {t}: FAILED {exc}")
            continue
        chunks = chunk_text(text, t)
        all_chunks.extend(chunks)
        print(f"[m6] {t}: {len(text)} chars -> {len(chunks)} chunks")
    if not all_chunks:
        raise SystemExit("[m6] no 10-K chunks fetched; corpus build failed")
    vec, mat = build_index(all_chunks)
    save_index(all_chunks, vec, mat)
    print(f"[m6] indexed {len(all_chunks)} chunks from {len(tickers)} tickers")
