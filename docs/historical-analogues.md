# Historical Analogues (workstream 2)

Product question: "When this ticker had a day like this before, what happened next?"

## Design

For a ticker and a query date, rank past news days by similarity to the query's
news profile. Similarity is computed on three features, all available with no
lookahead:

- `abnormal_z`: event-day return standardized against the trailing 60 trading days
- `sentiment`: mean article P(positive) mapped to [-1, 1]
- `volume`: log1p(number of articles)

Features are standardized across the ticker's event population; ranking is by
Euclidean distance. The query is excluded. Each analogue reports its distance,
per-feature deltas, top headline, and close-to-close forward returns at +1/+5/+10
trading days. Aggregates are median, mean, and fraction positive per horizon.

Framing is deliberately descriptive: "what happened next", never a forecast.

## Key design decision: events are per-pub-date, not windowed

The first implementation defined an event as "a trading day with scored news in
its [date-1d, date+1d] window", mirroring the explain view. Testing caught a real
degeneracy: an article published on day d lands in the windows of d-1, d, and d+1,
so every event's nearest "analogue" was its own neighboring day sharing the same
articles (distance ~ 0). The fix: each article belongs to exactly one event, its
pub_date (snapped forward to the next trading day for weekend publication). The
query keeps the window convention for consistency with the explain view
("the news around this move"); candidates are historical news days.

A second honesty guard: analogues within 3 calendar days of the query date are
flagged as "likely the same news burst, not an independent episode".

## Loud failures

Unknown ticker, invalid date, date outside price coverage, query with no scored
news in its window, or no event population with enough trailing history for
abnormal_z all raise explicit errors; they never return an empty or fabricated
result. Analogues near the end of price coverage get truncated forward returns
with a note.

## Real-data spot check (MSFT @ 2026-01-29, the -10% AI-spending rout)

Query: 22 articles, mean P(positive) 0.1%, abnormal z -7.46. Top analogue:
2025-11-06 ("Microsoft Stock Falls Amid Surplus Chip ...", distance 3.98).
Forward medians across the top 5 analogues: +1d -1.6%, +5d -1.6%, +10d -6.7%.
Descriptive only.

## Entry points

- Python: `python -m signal_lab.analogues --ticker MSFT --date 2026-01-29 [--k 5] [--json]`
- API: `GET /api/analogues?ticker=MSFT&date=2026-01-29`
- Dashboard: "Historical analogues" panel under "Why did it move?"
