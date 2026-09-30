# signallab

Signal Lab is a news-sentiment market event engine: it ingests financial
news and market prices, extracts company entities, scores sentiment with
classical ML, tests whether sentiment spikes move prices (event study),
backtests signal strategies walk-forward, and explains signals with a
citation-grounded RAG layer over 10-K filings.

The `signallab` CLI installs everything and opens a **browser UI that is the
interface**: pipeline controls, a live terminal-style console, and
presentation-ready charts. Built as interview preparation for a Data
Scientist role, where a live run is the demo.

## Install

```bash
npm install -g signallab
```

Requires Node 18+ and Python 3.10+.

## Usage

```bash
signallab            # set up if needed, start server, open the browser UI
signallab demo       # same as above
signallab run <stage># run one stage in the terminal (ingest|nlp|models|stats|backtest|rag)
signallab seed       # load the bundled demo snapshot (works offline)
signallab setup      # create the Python venv and install the engine deps
signallab doctor     # check node, python, engine venv, data dir
signallab serve       # start the server without opening a browser
signallab --help
```

Options: `--port <n>` (default 3100), `--no-open` (don't open a browser).

## The demo flow

1. `signallab` opens http://localhost:3100.
2. Press **Run full demo**: the six stages run in order, logs stream into
   the console pane, and each visual fills in as its stage completes.
3. No network? Press **Load demo snapshot** first: prebuilt data renders
   every visual instantly, and stages can still be re-run live afterward.

## Architecture

```
signallab (bin/signallab.js)
  ├─ setup  → engine/.venv + pip install engine/requirements.txt
  ├─ doctor → node / python3 / venv / data-dir checks
  └─ demo   → express server (server.js) + open browser
                 ├─ public/index.html  (dashboard: pipeline cards, console, charts)
                 ├─ /api/run/:stage    (spawns engine .venv python -m signal_lab.&lt;stage&gt;.run, SSE logs)
                 ├─ /api/results/:stage(cached stage JSON)
                 └─ /api/seed         (loads bundled demo snapshot)
engine/
  ├─ signal_lab/      (Python engine: ingest, nlp, models, stats, backtest, rag)
  ├─ requirements.txt
  └─ seed_data/       (aliases, labels, demo DuckDB + artifacts for offline demo)
data/                 (runtime: DuckDB, artifacts, results; created on demand)
```

Stage runners print a single JSON summary to stdout (logs go to stderr).
The server never fabricates results: a failed stage surfaces its stderr tail
in the UI console and API responses.

## Charts

Chart.js is vendored in `public/vendor/`; the UI works without CDN access.

## Publishing

```bash
npm pack    # inspect the tarball
npm publish # needs npm auth; run as the package owner
```
