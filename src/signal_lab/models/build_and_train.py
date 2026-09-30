"""Build the M3 weak-label dataset and train the sentiment classifier.

Dataset: each news article -> (text, ticker, published_date, label) where
the label comes from the configured weak-labeling scheme in
``signal_lab.models.labels``:

- ``windowed`` (0.3.0 default): 1 if the ticker's cumulative abnormal
  return vs the market over trading days t+1..t+window_days lands in the
  top decile, else 0; articles only survive the attention filter when
  their ticker-day carried enough news presence.
- ``baseline`` (the 0.2.0 scheme): 1 if the ticker's next-trading-day
  abnormal return vs SPY lands in the top decile, else 0. Kept so old vs
  new labeling stays measurable.

Saves:
  data/artifacts/vectorizer.pkl, model.pkl (joblib)
  data/artifacts/scored_news.csv (url, ticker, published_at, proba for M4/M6)

Usage: python -m signal_lab.models.run [--label-scheme windowed|baseline ...]
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from signal_lab.ingest import DB_PATH, fetch_prices, store_prices
from signal_lab.models import (
    feature_names,
    featurize,
    make_text,
    report_table,
    train,
)
from signal_lab.models.context_features import (
    CONTEXT_FEATURE_NAMES,
    build_context_features,
)
from signal_lab.models.labels import LabelConfig, build_labels
from signal_lab.models.lexicon import LEXICON_FEATURE_NAMES
from signal_lab.nlp import extract_baseline

REPO_ROOT = Path(__file__).resolve().parents[3]
ART = REPO_ROOT / "data" / "artifacts"


def build_dataset(
    log=print, label_cfg: LabelConfig | None = None
) -> tuple[pd.DataFrame, dict]:
    """Build the labeled article frame. Returns (df, label_config block)."""
    label_cfg = (label_cfg or LabelConfig()).validated()
    con = duckdb.connect(DB_PATH, read_only=True)
    news = con.execute(
        "SELECT url, title, body_snippet, published_at FROM news_raw WHERE title <> ''"
    ).fetchdf()
    prices = con.execute("SELECT ticker, date, close FROM prices_daily").fetchdf()
    con.close()
    prices["date"] = pd.to_datetime(prices["date"]).dt.date

    # Ensure the market benchmark exists. (Date strings must be plain
    # YYYY-MM-DD: yfinance rejects datetime strings with a time component.)
    if label_cfg.benchmark not in set(prices["ticker"]):
        log(f"[m3] fetching {label_cfg.benchmark} benchmark prices")
        bench = fetch_prices(
            [label_cfg.benchmark],
            prices["date"].min().isoformat(),
            prices["date"].max().isoformat(),
        )
        store_prices(bench)
        con = duckdb.connect(DB_PATH, read_only=True)
        prices = con.execute("SELECT ticker, date, close FROM prices_daily").fetchdf()
        con.close()
        prices["date"] = pd.to_datetime(prices["date"]).dt.date

    news["pub_date"] = pd.to_datetime(news["published_at"], utc=True).dt.date

    # Ticker per article via the deterministic baseline extractor.
    # Match on title + snippet: GDELT often returns the company mention in
    # the body rather than the headline.
    tickers, dropped = [], 0
    news["text"] = (
        news["title"].fillna("") + " " + news["body_snippet"].fillna("")
    ).str.strip()
    for text in news["text"]:
        hits = extract_baseline(text)
        if hits:
            tickers.append(max(hits, key=lambda h: h[2])[1])
        else:
            tickers.append(None)
            dropped += 1
    news["ticker"] = tickers
    news = news.dropna(subset=["ticker"]).reset_index(drop=True)
    log(f"[m3] articles with a ticker: {len(news)} (dropped {dropped} with none)")

    # 0.3.0 WS3: market-context features. Every feature is computed from
    # data strictly before the article's publish date (no leakage; see
    # signal_lab.models.context_features). Rows whose ticker lacks enough
    # pre-publication price history are dropped here so the label_config
    # block below still describes exactly the rows that get labeled.
    ctx = build_context_features(news, prices, benchmark=label_cfg.benchmark, log=log)
    ctx_ok = ~ctx[CONTEXT_FEATURE_NAMES].isna().any(axis=1)
    log(
        f"[m3] context features: dropped {int((~ctx_ok).sum())} rows "
        "with insufficient pre-publication price history"
    )
    news = news[ctx_ok].reset_index(drop=True)
    ctx = ctx[ctx_ok].reset_index(drop=True)

    labeled, label_block = build_labels(news, prices, label_cfg)
    log(
        f"[m3] scheme={label_block['scheme']} window={label_block['window_days']}d "
        f"attention={label_block['attention']['enabled']} "
        f"labeled rows: {label_block['n_labeled']}, "
        f"positive rate: {label_block['positive_rate']:.3f}, "
        f"top-decile cutoff: {label_block['cutoff']:+.4f}"
    )
    labeled["text"] = news.set_index("url").loc[labeled["url"], "text"].values
    labeled = labeled.merge(ctx[["url"] + CONTEXT_FEATURE_NAMES], on="url", how="left")
    cols = [
        "url",
        "text",
        "ticker",
        "published_at",
        "pub_date",
        "fwd_days",
        "n_articles",
        "ticker_fwd_ret",
        "mkt_fwd_ret",
        "abn_ret",
        "label",
    ] + CONTEXT_FEATURE_NAMES
    return labeled[cols].reset_index(drop=True), label_block


def _column_std(X) -> np.ndarray:
    """Per-column std, sparse-safe (scipy sparse matrices have no .std)."""
    if hasattr(X, "toarray"):
        mean = np.ravel(np.asarray(X.mean(axis=0), dtype=float))
        mean_sq = np.ravel(np.asarray(X.power(2).mean(axis=0), dtype=float))
        return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    return np.ravel(np.asarray(np.std(X, axis=0), dtype=float))


def feature_importance_block(model_name, model, names, dense_names, X) -> dict:
    """Per-feature importance for the persisted (best) model.

    This is the stage-3 ``feature_importance`` JSON block. Linear models
    use |coefficient| * feature-std (standardized magnitude, so raw-scale
    differences between TF-IDF, lexicon, and context features do not
    dominate the ranking); tree models use their own feature_importances_.
    Reports the top 20 text terms plus every dense (lexicon + context)
    feature, each ranked by importance.
    """
    if hasattr(model, "coef_"):
        coef = np.ravel(np.asarray(model.coef_, dtype=float))
        importance = np.abs(coef) * _column_std(X)
        method = "abs_coef_times_std"
    elif hasattr(model, "feature_importances_"):
        importance = np.ravel(np.asarray(model.feature_importances_, dtype=float))
        method = "feature_importances_"
    else:
        return {
            "model": model_name,
            "method": "unavailable",
            "n_features": len(names),
            "top_text_terms": [],
            "dense_features": [],
        }
    dense_set = set(dense_names)
    top_text, dense_rows = [], []
    for i in np.argsort(importance, kind="stable")[::-1]:
        entry = {
            "feature": names[i],
            "importance": round(float(importance[i]), 6),
        }
        if names[i] in dense_set:
            dense_rows.append(entry)
        elif len(top_text) < 20:
            top_text.append(entry)
    dense_rows.sort(key=lambda d: d["importance"], reverse=True)
    return {
        "model": model_name,
        "method": method,
        "n_features": len(names),
        "top_text_terms": top_text,
        "dense_features": dense_rows,
    }


def run_pipeline(
    log=print, label_cfg: LabelConfig | None = None, finbert: bool = False
) -> dict:
    """Build dataset, train, persist artifacts. Returns the train() result,
    with a 'label_config' block describing the labeling used and a
    'finbert_ablation' block (0.3.0 workstream 2; opt-in via ``finbert=True``
    or the SIGNAL_LAB_FINBERT=1 env var, offline-safe)."""
    label_cfg = (label_cfg or LabelConfig()).validated()
    ART.mkdir(parents=True, exist_ok=True)
    df, label_block = build_dataset(log=log, label_cfg=label_cfg)
    ctx_df = df[CONTEXT_FEATURE_NAMES]
    result = train(df, text_col="text", dense_extra=ctx_df)
    table = report_table(result)
    log(table.to_string(index=False))
    log(
        f"n_train={result['n_train']} n_test={result['n_test']} "
        f"positive_rate={result['positive_rate']:.3f}"
    )

    # Persist the best model (highest test PR-AUC among fitted models).
    scored = [
        (r.pr_auc, r.name) for r in result["reports"] if r.name in result["models"]
    ]
    best_name = max(scored)[1]
    joblib.dump(result["vectorizer"], ART / "vectorizer.pkl")
    joblib.dump(result["models"][best_name], ART / "model.pkl")
    (ART / "best_model.txt").write_text(best_name)
    log(f"[m3] saved best model: {best_name}")

    # Score every article for M4/M6.
    vec = result["vectorizer"]
    model = result["models"][best_name]
    proba = model.predict_proba(featurize(make_text(df), vec, dense_extra=ctx_df))[:, 1]
    scored_df = df[["url", "text", "ticker", "published_at", "pub_date"]].copy()
    scored_df["proba"] = proba
    scored_df.to_csv(ART / "scored_news.csv", index=False)
    log(f"[m3] scored {len(scored_df)} articles -> {ART / 'scored_news.csv'}")
    result["best_model"] = best_name
    result["label_config"] = label_block
    # 0.3.0 WS2: FinBERT ablation (opt-in; falls back cleanly offline).
    result["finbert_ablation"] = maybe_run_finbert_ablation(
        df, log=log, tfidf_result=result, requested=finbert
    )
    # 0.3.0 WS3: per-feature importance for the persisted model. Uses the
    # same feature space the model was trained on (text + lexicon + ctx).
    result["feature_importance"] = feature_importance_block(
        best_name,
        model,
        feature_names(vec, CONTEXT_FEATURE_NAMES),
        list(LEXICON_FEATURE_NAMES) + list(CONTEXT_FEATURE_NAMES),
        featurize(make_text(df), vec, dense_extra=ctx_df),
    )
    return result


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 0.3.0 workstream 2: optional feature builders (append-only section).
#
# FEATURE_BUILDERS maps a feature-set name to
# ``builder(texts, vectorizer, **kwargs) -> feature matrix``. The default
# pipeline and workstream 1 use "tfidf_lexicon". The sibling workstream 3
# (market-context features) appends its own builder to this dict; keep all
# additions here so the later merge stays trivial.
# ---------------------------------------------------------------------------

FEATURE_BUILDERS: dict = {
    "tfidf_lexicon": lambda texts, vectorizer, **kwargs: featurize(texts, vectorizer),
}


def register_feature_builder(name: str, builder) -> None:
    """Register an additional feature builder (idempotent by name)."""
    if name in FEATURE_BUILDERS:
        raise ValueError(f"feature builder {name!r} is already registered")
    FEATURE_BUILDERS[name] = builder


def _register_finbert_builders() -> None:
    # Imported lazily: embeddings.py never imports torch/transformers at
    # module scope, so this stays offline-safe.
    from signal_lab.models.embeddings import (
        combined_feature_matrix,
        finbert_feature_matrix,
    )

    def _finbert_only(texts, vectorizer, **kwargs):
        return finbert_feature_matrix(texts, **kwargs)

    def _combined(texts, vectorizer, **kwargs):
        return combined_feature_matrix(texts, vectorizer, **kwargs)

    for _name, _fn in (
        ("finbert_only", _finbert_only),
        ("tfidf_lexicon_finbert", _combined),
    ):
        if _name not in FEATURE_BUILDERS:
            FEATURE_BUILDERS[_name] = _fn


_register_finbert_builders()


def _finbert_requested(explicit: bool) -> bool:
    import os

    return bool(explicit) or os.environ.get("SIGNAL_LAB_FINBERT", "").strip() == "1"


def maybe_run_finbert_ablation(
    df, log=print, tfidf_result: dict | None = None, requested: bool = False
) -> dict:
    """Run the FinBERT ablation when requested, else a not-requested block.

    Always returns a JSON-safe ``finbert_ablation`` block; never raises for
    a missing torch/transformers install (the fallback path logs and
    reports TF-IDF-only).
    """
    from signal_lab.models.embeddings import FINBERT_MODEL_ID, run_finbert_ablation

    if not _finbert_requested(requested):
        return {
            "available": False,
            "model": FINBERT_MODEL_ID,
            "reason": "not requested (pass --finbert or set SIGNAL_LAB_FINBERT=1)",
        }
    try:
        return run_finbert_ablation(
            df, text_col="text", log=log, tfidf_result=tfidf_result
        )
    except Exception as exc:  # noqa: BLE001 - ablation must never break stage 3
        log(f"[m3] finbert ablation failed ({exc}); continuing TF-IDF-only")
        return {
            "available": False,
            "model": FINBERT_MODEL_ID,
            "reason": f"ablation error: {exc}",
        }
