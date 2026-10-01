# Negative result: news sentiment over price-only (two gate runs)

Date: 2026-10-01. This document is the pre-specified FALLBACK for the
gate-2 retest program (docs/GATE_2_PREREGISTRATION.md). It covers both
gate runs honestly: what was tested, what the verdicts were, what
blocked the second run, and what was NOT tested.

## Gate 1 (completed 2026-10-01): NO-GO

- Scope: 693 dev rows, 10 tickers (AAPL, AMZN, BAC, GOOGL, JPM, META,
  MSFT, NVDA, TSLA, XOM), news t0 range 2025-11-24..2026-06-26, dev
  period up to 2026-06-30.
- Text representations tried: TF-IDF (5,000 features) + LM lexicon
  sentiment, fit per fold on train only.
- Primary comparison (variant C text+price vs variant A price-only,
  logreg_balanced, 5 walk-forward folds, purge + 5-day embargo):
  mean paired PR-AUC difference +0.0035, 95% CI [-0.0091, 0.0160],
  paired t p = 0.48, Wilcoxon p = 1.00. Verdict: NO-GO.
- Secondary findings: text-only variant lost to price-only in every
  fold; calibration and abstention experiments failed; no evidence the
  early-fold price signal was harness leakage (Step 2 leakage check:
  the early-fold signal is noise).
- The test period (2026-07-01..2026-08-31) and confirmation period
  (t0 >= 2026-09-01) were never evaluated.

## Gate 2 (pre-registered retest, NOT completed)

The pre-registered gate-2 plan (FinBERT as primary text arm, TF-IDF
also reported, widened to 101 tickers / ~3 years, same frozen
methodology) completed Steps 0-4:

- STEP 0: pre-registration written (docs/GATE_2_PREREGISTRATION.md).
- STEP 1: widened universe of 101 liquid US tickers built.
- STEP 2: leakage check on the early-fold price signal -> noise, no
  harness leakage found.
- STEP 3: widening code built and tested.
- STEP 4: FinBERT arm (ProsusAI/finbert, 768-d CLS embeddings) added
  as primary variant Fb/Fc.

STEP 5 (news ingest for the widened universe) was blocked:

1. GDELT DOC API hard throttle: HTTP 429 on effectively every
   request; projected ~100 hours for the full widening. Ingest
   checkpointed at 14 ok / 11 error quarters.
2. Bulk-source switch (user-authorized, one time-boxed session):
   - STEP 0a integrity check: the SPY price gap (prices missing before
     2024-10-15, backfilled later) had ZERO effect on gate 1. Rebuilt
     the original 909-row pre-widening frame with SPY truncated to
     pre-backfill state vs full backfilled SPY: 0 differing cells
     across all 49 columns (labels, all features including
     ctx_sector_rel5). Gate-1 results stand exactly as reported.
     (scripts/gate2_step0_spy_check.py, scripts/gate2_step0_spy_fulldiff.py)
   - STEP 1 access verification: GDELT bulk GKG files verified working
     (free, no auth, no documented rate limit, HTTPS, predictable
     15-min slot URLs). Schema verified: 27 tab-separated columns;
     the Extras field carries PAGE_PRECISEPUBTIMESTAMP (exact
     publication time) and PAGE_TITLE (article title). Timestamp
     mapping: use PAGE_PRECISEPUBTIMESTAMP for availability time,
     falling back to the slot DATE (15-min batch stamp). License:
     unlimited unrestricted use for any purpose without fee,
     attribution to the GDELT Project required. Article-title text is
     available; full article body text is not in the bulk files.
     BUT: measured throughput from this environment was ~1.4-2.2 MB/s
     aggregate, projecting the pre-registered 3-year pull (~300+ GB)
     at 38-78 hours. Infeasible inside the session time-box. The
     BigQuery public dataset (gdelt-bq.gdeltv2.gkg, free 1 TB/mo
     query tier) would solve this with server-side filtering, but
     requires GCP credentials this session did not have.
   - Per the pre-specified hard stop (STEP 1: stop after a few hours if
     neither source works in-session), the bulk path was not pursued.

## Deviations from pre-registration (logged)

- Gate-1 ingest used max_records=100 per request instead of the
  pre-registered 250 (silent API-side change in late September 2026;
  records were truncated to the first 100). Logged 2026-10-01.
- The news-source switch (GDELT DOC API -> bulk GKG files) was a
  user-authorized deviation after Step 5 was blocked by throttling.
  It was not completed: the source switch is verified as technically
  sound but not yet executed at the pre-registered scale.

## What was NOT tested

- The widened gate-2 benchmark itself (FinBERT primary on 101
  tickers / 3 years): never ran. No GO/NO-GO verdict exists for it.
- The test period (2026-07-01..2026-08-31) and confirmation period
  (t0 >= 2026-09-01) remain frozen and untouched.
- Phases 4-7 of the research program were never built.
- The bulk-GKG ingest pipeline (download, ticker matching,
  title-based text arms) was verified at small scale only.

## Verdict

Gate 1 stands as the only completed gate: NO-GO. The gate-2 retest
could not be executed within the authorized session; its pre-registered
code (FinBERT arm, widening, leakage-checked benchmark) remains on
main, ready for a future run with working news access.

Recommended next steps (user decision required): provide GCP
credentials so the bulk ingest can run through BigQuery with
server-side filtering (the clean solution: ~1-2 GB of query results
instead of ~300+ GB of downloads); or run the GKG bulk pull on a
machine with a faster pipe; or accept gate 1's NO-GO as the final
verdict.
