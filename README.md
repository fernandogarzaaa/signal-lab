# Signal Lab

A news-sentiment market event engine, built as interview preparation for a
Data Scientist role at LSEG (London Stock Exchange Group), recruited via
Globetec Solutions for a new Data Science team in Taguig, Metro Manila.

## Why this project

The role asks for Python, NLP, LLMs/GenAI, RAG, SQL, PySpark, and ML/DL, with
responsibilities across data pipelines, web scraping, entity extraction,
document processing, model training/deployment, MLOps, and CI/CD.

LSEG's actual data-science work looks like this:

- **News Analytics**: NLP measuring company sentiment, relevance, and novelty
  across 40,000 companies (their MarketPsych NLP engine).
- **Entity resolution**: NER on unstructured text linked to PermID identifiers,
  building knowledge graphs (60k projects, 24k entities, 4M+ relations).
- **Event impact**: ML models that detect when news clusters move currency and
  equity prices (FX Impact Intelligence, Mosaic anomaly explanation).
- **Workspace AI**: conversational search and Deep Research combining LSEG data
  with foundation models.

Signal Lab is a small, honest version of that stack. Each milestone targets one
of the gaps behind the interview: classical statistics, classical ML theory,
and experimental design, while using your strengths (NLP, RAG, agents, data
processing) as the delivery vehicle.

## The build

```
news + prices  -->  entity extraction  -->  sentiment classifier  -->
event study  -->  walk-forward backtest  -->  RAG explainer
```

| # | Milestone | Gap it fills | Interview question it answers |
|---|-----------|--------------|------------------------------|
| M1 | News + price data pipeline | data engineering | "Tell us about analysing a large dataset" |
| M2 | Entity extraction (PermID-lite) | NLP fundamentals | "How would you link entities across documents?" |
| M3 | Sentiment classifier, classical ML | classical ML theory | "Build a model on a highly imbalanced dataset" |
| M4 | Event study: do sentiment spikes move prices? | classical statistics | "How will you incorporate Statistics into Data Science?" |
| M5 | Walk-forward backtest with significance testing | experimental design | "How do you validate a model without fooling yourself?" |
| M6 | RAG explainer over filings/transcripts | applied GenAI | project walkthrough differentiator |
| M7 | Timed Python live-coding drills | interview format | the actual first-round format |

Full specs and acceptance criteria: [MILESTONES.md](MILESTONES.md).
Python drills: [docs/DRILLS.md](docs/DRILLS.md).

## Ground rules

- Free data only: GDELT news API (no key), Yahoo Finance prices (yfinance),
  SEC EDGAR filings. No paid APIs.
- Every milestone ends with a short writeup in `docs/`. The writeups become
  your interview talking points: problem, data, approach, metrics, deployment.
- No lookahead bias, ever. Timestamps are sacred in M4 and M5.
- Skeletons in `src/` are yours to implement. Each module docstring states its
  contract and acceptance criteria.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m signal_lab.ingest.run --help
pytest tests/
```

## Status

Scaffold only. Milestones M1-M7 are unimplemented and waiting to be built.
