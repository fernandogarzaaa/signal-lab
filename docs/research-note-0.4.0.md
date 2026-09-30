# Signal Lab 0.4.0 Research Note

**Date:** 2026-10-01
**Question:** Can the model actually predict? We tried three things.

## What we tried

1. **Windowed labels** (3-day CAR + attention filter) vs baseline (1-day top-decile)
2. **Novelty features** (FinBERT similarity to trailing-7d same-ticker coverage)
3. **FinBERT representations** (768-dim frozen embeddings + 7 market-context features)

## Results (purged walk-forward, 902 rows, 5 folds)

| Representation | Mean PR-AUC | 95% CI | Mean F1 |
|---|---|---|---|
| TF-IDF + lexicon + ctx (0.3.0) | 0.1219 | [0.027, 0.217] | - |
| **FinBERT + ctx (0.4.0)** | **0.1502** | **[0.072, 0.229]** | **0.189** |
| Chance (prevalence) | 0.1131 | - | - |

FinBERT+ctx beats TF-IDF by +0.028 (+23% relative) with a higher CI lower bound.
The win is consistent: it beat TF-IDF in 4 of 5 folds.

| Label scheme | Mean PR-AUC (3-fold) | Verdict |
|---|---|---|
| Baseline (1-day) | 0.1412 | **Keep** |
| Windowed (3-day CAR + attention) | 0.0882 | **Reject**: adds noise |

| Feature | Delta PR-AUC | Verdict |
|---|---|---|
| + Novelty (3 features) | +0.0042 | **Reject**: noise, not signal |

## What this means

The model got smarter. FinBERT embeddings capture financial semantics that
TF-IDF bag-of-words misses. The 7 market-context features (volatility,
momentum, sector-relative, source tier, word count, hour) add value on top.

But the CIs are still wide. With 902 rows and 10% prevalence, we're measuring
a weak signal through heavy noise. The binding constraint is label quality,
not representation.

## Negative results are results

- Windowed multi-day labels: the attention filter and longer window add more
  noise than signal. Baseline stays.
- Novelty features: semantic similarity to recent coverage doesn't predict
  next-day moves in this sample. LSEG has novelty metadata; ours is open and
  reproducible, but it doesn't help the model. Documented, not shipped.
- GDELT history expansion: blocked by 429 rate limits. SPY backfill succeeded
  (902 rows, real sector-relative feature).

## Next: the gold set

The interactive `gold_set label` command is ready. Human labels on 150
articles will quantify the weak-label noise directly. That's the experiment
that matters now.
