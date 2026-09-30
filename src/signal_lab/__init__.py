"""Signal Lab: news-sentiment market event engine.

Milestone map:
    ingest   M1  news (GDELT) + prices (yfinance) -> DuckDB
    nlp      M2  entity extraction, company mention -> ticker
    models   M3  classical sentiment classifier, imbalanced data
    stats    M4  event study: do sentiment spikes move prices?
    backtest M5  walk-forward strategy comparison, no lookahead
    rag      M6  cited explanations over filings and transcripts
"""

import math


def sanitize_json(obj):
    """Recursively convert NaN/Inf floats to None so json.dumps output
    is strict JSON (parseable by JavaScript's JSON.parse)."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    return obj


def write_docs_csv(df, filename, log=print):
    """Write a repo docs/ CSV artifact. In the packaged npm distribution
    there is no docs/ directory, so create it if possible and skip with
    a warning if not. Never fails the pipeline stage."""
    from pathlib import Path
    target = Path(__file__).resolve().parent.parent.parent / "docs" / filename
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target, index=False)
    except OSError as exc:
        log(f"[warn] skipping docs artifact {filename}: {exc}")
