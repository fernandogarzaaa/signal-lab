# Signal Lab

A news-sentiment market event engine, built as interview preparation for a
Data Scientist role at LSEG (London Stock Exchange Group), recruited via
Globetec Solutions for a new Data Science team in Taguig, Metro Manila.

It reads financial news, checks whether markets reacted, tests whether those
reactions form a usable trading signal, and explains what it finds with
citations. Packaged as the `signallab` npm CLI with a live browser dashboard.

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

## How it works

Six stages, run in order. Every stage reads from and writes to a local
DuckDB database, prints strict JSON to stdout, and is idempotent (reruns
change nothing).

```
news + prices  -->  entity extraction  -->  sentiment classifier  -->
event study  -->  walk-forward backtest  -->  RAG explainer
```

| # | Stage | What it does |
|---|-------|--------------|
| 1 | Ingest | Pulls news from the GDELT 2.1 DOC API (no key) and daily OHLCV prices via yfinance into DuckDB. Upsert on natural keys, never duplicates. |
| 2 | NLP / entity extraction | Resolves company mentions in article text to tickers (rules + optional spaCy NER). |
| 3 | Sentiment classifier | Trains a logistic regression on TF-IDF plus Loughran-McDonald finance-lexicon features. Weak labels come from the market itself: label 1 if the ticker's next-day abnormal return vs SPY lands in the top decile (~10% positive rate). Compares candidates (plain, class weights, resampling, threshold tuning, LightGBM) on a time-based split and reports PR-AUC, the honest metric for imbalanced data. |
| 4 | Event study | Classical event study: do sentiment spikes coincide with abnormal returns? Reports cumulative abnormal returns with significance tests. |
| 5 | Backtest | Walk-forward backtest of a sentiment-driven strategy vs buy-and-hold, with Sharpe ratios and significance testing. Built to not fool itself. |
| 6 | RAG explainer | Cited retrieval over SEC EDGAR 10-K filings: explains detected events with quoted, sourced passages. |

Architecture: a Python analytical engine (`src/signal_lab/`), a Node/npm CLI
and Express dashboard (`packages/cli/`), local DuckDB storage, server-sent
events for live logs, and vendored Chart.js. A seeded demo snapshot ships in
the package so `signallab seed` works fully offline; it never replaces the
real live pipeline.

Full specs and acceptance criteria: [MILESTONES.md](MILESTONES.md).
Model details: [docs/model-card.md](docs/model-card.md).
Known limitations: [docs/limitations.md](docs/limitations.md).
Interview talking points: [docs/interview-cheat-sheet.md](docs/interview-cheat-sheet.md).
Python drills: [docs/DRILLS.md](docs/DRILLS.md).

## Ground rules

- Free data only: GDELT news API (no key), Yahoo Finance prices (yfinance),
  SEC EDGAR filings. No paid APIs.
- No lookahead bias, ever. Timestamps are sacred in stages 4 and 5.
- No stubs or TODOs in shipped code; CI audits for both.
- Honest numbers. When the model is bad, the docs say the model is bad.

## Quick start

```bash
npm install -g signallab
signallab            # set up if needed, start server, open the browser UI
signallab run models # run one stage in the terminal
signallab seed       # load the bundled demo snapshot (works offline)
signallab doctor     # check node, python, engine venv, data dir
```

Or from source:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m signal_lab.ingest.run --tickers AAPL,MSFT,JPM --days 90
pytest tests/
```

## Status

**0.1.0** is published on npm (`signallab@0.1.0`): all six stages verified
end to end on 30 days of data (223 news rows, 681 price rows across
AAPL/MSFT/JPM), dashboard confirmed with real charts and cited RAG output.

**0.2.0** (in progress): Loughran-McDonald finance-lexicon features merged
(PR #2, CI green), plus a 365-day backfill scaling the training set ~5x
(784 news rows and 2,571 price rows across 10 tickers so far; an overnight
retry is filling the remaining GDELT windows). Honest results are reported
in the [model card](docs/model-card.md): on current data the classifier
scores PR-AUC ~0.16 against a ~0.10 chance level. Not good yet, and we say
so. The 0.3.0 roadmap ([docs/roadmap-0.3.0.md](docs/roadmap-0.3.0.md),
tracking issue #4) covers label quality, FinBERT features, market-context
features, calibration, and purged walk-forward validation.

## Data attribution

News via the GDELT Project 2.1 DOC API. Prices via Yahoo Finance (yfinance).
Filings via SEC EDGAR. Finance sentiment word lists derived from the
Loughran-McDonald Master Dictionary (Loughran & McDonald, 2011, *Journal of
Finance* 66:1, 35-65), used under its academic-research terms; see
`src/signal_lab/models/data/` for the full notice.
