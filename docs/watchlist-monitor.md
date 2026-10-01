# Watchlist monitor (workstream 3)

A near-real-time news monitor over the GDELT 2.1 DOC API. It watches a
user-managed ticker watchlist, stores fresh articles, scores them with the
shipped production sentiment model, and surfaces freshness and failures
honestly in the dashboard.

Positioning: this is a **cited news monitor, not a predictor**. It tells
you what the news is saying about your tickers right now, with a sentiment
score and price context. It never forecasts returns.

## How it works

```
poll cycle (every 15 min in the dashboard server, or on demand)
  1. for each watchlist ticker: GDELT DOC query for the trailing window
     (default 30 min, overlapping the 15-min GDELT refresh cadence)
  2. upsert new articles into DuckDB news_raw (idempotent, keyed by URL)
  3. refresh price context via yfinance (also feeds the price badges)
  4. attribute each article to a ticker (alias extractor; falls back to the
     queried ticker for tickers with no aliases on file)
  5. score with the production model (TF-IDF + lexicon + 7 market-context
     features, same feature space as training)
  6. append to data/artifacts/monitor_scored.csv; write poll health to
     data/artifacts/monitor_state.json
```

## Failure semantics (loud, never silent)

- A failed ticker query does not kill the poll; the failure is recorded in
  `monitor_state.json` and shown as a red badge in the dashboard.
- A failed price refresh does not kill news polling; it is recorded
  separately (`price_error`).
- A missing `vectorizer.pkl` / `model.pkl` / `best_model.txt` raises
  `FileNotFoundError` with an explicit message. There is deliberately no
  fallback: scoring with fabricated numbers would be worse than failing.
- Articles that cannot be scored honestly (insufficient pre-publication
  price history for the context features) are kept with an empty `proba`
  and `score_note = "unscored: ..."`, never a zero.
- The dashboard shows "not polled yet", "last checked Xm ago", a "stale"
  badge past 45 minutes, and per-ticker failure badges.

## Feeding the evidence views

`monitor_scored.csv` uses the same schema as `scored_news.csv`
(url, text, ticker, published_at, pub_date, proba) plus `score_note`.
`signal_lab.monitor.merge_scored()` folds it into the frame read by
`explain_move.load_news_prices()`, so both the "Why did it move?" view and
the historical analogues see freshly polled articles. Monitor rows win on
URL conflicts; unscored rows are dropped downstream (they have no proba).
The training artifact `scored_news.csv` itself is never mutated.

## Interface

CLI:

```
python -m signal_lab.monitor --poll [--tickers MSFT,NVDA] [--lookback-min 30]
python -m signal_lab.monitor --watchlist-list / --watchlist-add TSLA / --watchlist-remove TSLA
python -m signal_lab.monitor --watchlist-json   # dashboard snapshot
```

Dashboard API (packages/cli/server.js):

- `GET /api/watchlist` -> tickers, prices, recent articles, monitor state
- `POST /api/watchlist` {ticker} -> add (validated: 1-8 chars A-Z 0-9 . -)
- `DELETE /api/watchlist/:ticker` -> remove
- `POST /api/monitor/poll` -> 202 immediately; the poll runs in the background
  (a cycle takes minutes: GDELT politeness sleep + body fetch + scoring)

The server also polls in the background every `SIGNAL_LAB_MONITOR_MINUTES`
(default 15, matching the GDELT refresh cadence; 0 disables, first run 60s
after startup).

## Cost and rate notes

- GDELT DOC API is free but throttles aggressively: 30s politeness sleep
  between ticker queries, exponential backoff on 429 (inherited from the
  ingest module).
- A 10-ticker poll takes roughly 5 minutes wall-clock, comfortably inside
  the 15-minute cadence.
- Article body fetch is on by default (the model was trained on title +
  snippet); `--no-bodies` scores title-only.
