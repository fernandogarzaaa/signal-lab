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
  data/artifacts/scored_news.csv (url, ticker, published_at, proba for M4/M6;
    0.6.0: proba is the fold-test OUT-OF-SAMPLE score from purged
    walk-forward on dev; NaN for rows never in a test block)

Usage: python -m signal_lab.models.run [--label-scheme windowed|baseline ...]
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score

from signal_lab.ingest import DB_PATH, fetch_prices, store_prices
from signal_lab.models import (
    ModelReport,
    _metrics,
    _take_rows,
    feature_names,
    featurize,
    fit_challenger_candidates,
    make_text,
    report_table,
    train,
    tune_threshold,
)
from signal_lab.models.context_features import (
    CONTEXT_FEATURE_NAMES,
    build_context_features,
)
from signal_lab.models.labels import LabelConfig, apply_legacy_labeling, build_labels
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

    labeled, label_info = build_labels(news, prices, label_cfg)
    log(
        f"[m3] scheme={label_info['scheme']} window={label_info['window_days']}d "
        f"attention_filter=per-fold(train-only) "
        f"labeled rows: {label_info['n_rows']}, "
        f"too_recent_dropped={label_info['too_recent_dropped']}, "
        f"partial_window_dropped={label_info['partial_window_rows_dropped']}"
    )
    labeled["text"] = news.set_index("url").loc[labeled["url"], "text"].values
    labeled = labeled.merge(
        ctx[["url", "ctx_asof"] + CONTEXT_FEATURE_NAMES], on="url", how="left"
    )
    # Legacy global-cutoff labels, kept for the single-split train() path
    # and CLI compatibility. The validation framework (walk-forward,
    # final-train) does NOT use this column; it builds per-fold labels
    # from abn_ret via signal_lab.validation.labels (audit 4c/4d).
    legacy_labeled, label_block = apply_legacy_labeling(labeled, label_cfg)
    log(
        f"[m3] legacy label_config: positive_rate={label_block['positive_rate']:.3f}, "
        f"top-decile cutoff={label_block['cutoff']:+.4f} (global; validation uses per-fold)"
    )
    cols = [
        "url",
        "text",
        "ticker",
        "published_at",
        "pub_date",
        "t0",
        "t1",
        "fwd_days",
        "n_articles",
        "ticker_fwd_ret",
        "mkt_fwd_ret",
        "abn_ret",
        "label",
        "ctx_asof",
    ] + CONTEXT_FEATURE_NAMES
    out = legacy_labeled[cols].reset_index(drop=True)
    return out, label_block


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


def load_trading_calendar_days(log=print) -> np.ndarray:
    """Sorted unique trading-date ordinals (union over tickers).

    Used for trading-day embargo arithmetic in the validation framework.
    """
    import duckdb

    con = duckdb.connect(DB_PATH, read_only=True)
    dates = con.execute("SELECT DISTINCT date FROM prices_daily").fetchdf()["date"]
    con.close()
    ords = np.sort(
        pd.to_datetime(dates).dt.date.apply(lambda d: d.toordinal()).unique().astype(np.int64)
    )
    log(f"[m3] union trading calendar: {len(ords)} days")
    return ords


def _prepare_fit_frame(
    df: pd.DataFrame,
    train_pos: np.ndarray,
    valid_pos: np.ndarray | None,
    label_cfg: LabelConfig,
    train_label_col: str | None,
    log,
    context: str,
) -> dict:
    """Shared label/attention/dedupe prep for selection and final-train.

    Labels are built per-fold-style: the cutoff comes from the TRAIN
    abnormal returns only, the attention quantiles from train rows only
    (audit 4c/4d). ``valid_pos=None`` means final-train (no eval block).
    Same-ticker same-t0 rows are down-weighted (1/group size).
    """
    from signal_lab.validation import dedupe
    from signal_lab.validation import labels as vlabels

    train_df = df.iloc[train_pos]
    if valid_pos is not None:
        full = df.iloc[np.concatenate([train_pos, valid_pos])]
        mask = vlabels.attention_keep_mask(
            train_df,
            full,
            min_articles=label_cfg.attention_min_articles,
            top_quartile=label_cfg.attention_top_quartile,
        ).to_numpy()
        tri_k = train_pos[mask[: len(train_pos)]]
        va_k: np.ndarray | None = valid_pos[mask[len(train_pos):]]
    else:
        mask = vlabels.attention_keep_mask(
            train_df,
            train_df,
            min_articles=label_cfg.attention_min_articles,
            top_quartile=label_cfg.attention_top_quartile,
        ).to_numpy()
        tri_k = train_pos[mask]
        va_k = None
    if len(tri_k) == 0:
        raise ValueError(f"[m3] {context}: attention filter emptied the train block")
    cutoff = vlabels.fold_cutoff(df["abn_ret"].iloc[tri_k], label_cfg.quantile)
    if train_label_col is not None:
        if train_label_col not in df.columns:
            raise ValueError(f"train_label_col={train_label_col!r} not in df columns")
        ytr_all = df[train_label_col].to_numpy(dtype=float)
        keep = ~np.isnan(ytr_all[tri_k])
        base = tri_k[keep]
        if len(base) == 0:
            raise ValueError(f"[m3] {context}: no usable training rows after abstention exclusion")
        ytr = ytr_all[base].astype(int)
    else:
        base = tri_k
        ytr = vlabels.fold_labels(df["abn_ret"].iloc[base], cutoff).to_numpy()
    if len(np.unique(ytr)) < 2:
        raise ValueError(f"[m3] {context}: fewer than 2 classes in the train block")
    sample_weight = dedupe.sample_weights(df.iloc[base])
    yva = (
        vlabels.fold_labels(df["abn_ret"].iloc[va_k], cutoff).to_numpy()
        if va_k is not None and len(va_k)
        else None
    )
    log(
        f"[m3] {context}: train={len(base)} valid={0 if yva is None else len(yva)} "
        f"cutoff={cutoff:+.4f} (train-only) pos_rate={ytr.mean():.3f}"
    )
    return {
        "tri_eff": base,
        "ytr": ytr,
        "va_pos": va_k,
        "yva": yva,
        "sample_weight": sample_weight,
        "cutoff": float(cutoff),
    }


def run_pipeline(
    log=print,
    label_cfg: LabelConfig | None = None,
    finbert: bool = False,
    train_labels: str = "weak",
    jev_labels_path: str | None = None,
) -> dict:
    """Build dataset, select, train, persist artifacts. Returns a result
    dict with the validation-framework blocks, plus a 'finbert_ablation'
    block (0.3.0 workstream 2; opt-in via ``finbert=True`` or the
    SIGNAL_LAB_FINBERT=1 env var, offline-safe).

    0.6.0 protocol (Phase 1; audit findings 4b/4c/4d/4f fixed):

    1. SELECTION on a single purged validation split of the development
       period (never on a test split): every challenger candidate is fit
       on the selection-train block and ranked by validation PR-AUC.
    2. FINAL TRAIN on all development rows purged against the untouched
       test period; the winner is refit and persisted as model.pkl /
       vectorizer.pkl.
    3. OUT-OF-SAMPLE scores from a purged walk-forward loop on dev: the
       ``proba`` column of scored_news.csv now holds fold-test OOS
       values (NaN for rows never in a test block: block 0, the test
       period, and confirmation). Previously it held in-sample scores
       from the selected model, which was optimistic by construction.

    train_labels selects the training-label source: "weak" (default,
    price-derived, reproducible), "jev" or "jev-conf06" (Jev-judged,
    opt-in; needs the Jev label file at jev_labels_path). Test metrics
    are always computed against weak labels.
    """
    from signal_lab.validation import periods, point_in_time
    from signal_lab.validation import splits as vsplits

    label_cfg = (label_cfg or LabelConfig()).validated()
    ART.mkdir(parents=True, exist_ok=True)
    df, label_info = build_dataset(log=log, label_cfg=label_cfg)
    point_in_time.check_frame(df, "run_pipeline")
    trading_days = load_trading_calendar_days(log=log)
    ctx_df = df[CONTEXT_FEATURE_NAMES]
    texts = make_text(df)

    train_label_col = None
    train_labels_block: dict = {"source": train_labels}
    if train_labels != "weak":
        from signal_lab.models.train_labels import (
            DEFAULT_JEV_LABELS_PATH,
            TRAIN_LABEL_COL,
            apply_train_labels,
        )

        effective_jev_path = jev_labels_path or DEFAULT_JEV_LABELS_PATH
        df = apply_train_labels(
            df, train_labels, jev_path=effective_jev_path, log=log
        )
        ctx_df = df[CONTEXT_FEATURE_NAMES]
        texts = make_text(df)
        train_label_col = TRAIN_LABEL_COL
        train_labels_block["jev_labels_path"] = str(effective_jev_path)
        train_labels_block["test_labels"] = "weak (fixed evaluation target)"

    vw_config = vsplits.WalkForwardConfig(
        n_splits=2,
        horizon_days=label_cfg.window_days,
        embargo_days=label_cfg.window_days + 2,
    )

    # 1. SELECTION on the purged validation split (audit 4b).
    sp = vsplits.single_purged_split(df, vw_config, trading_days)
    log(
        f"[m3] selection split: valid {sp['valid_start']}..{sp['valid_end']} "
        f"(purged {sp['n_purged']} train rows)"
    )
    sel = _prepare_fit_frame(
        df, sp["train_idx"], sp["valid_idx"], label_cfg, train_label_col, log,
        "selection",
    )
    vec_sel = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True
    )
    vec_sel.fit(texts.iloc[sel["tri_eff"]])
    Xtr_sel = featurize(
        texts.iloc[sel["tri_eff"]], vec_sel,
        dense_extra=_take_rows(ctx_df, sel["tri_eff"]),
    )
    Xva_sel = featurize(
        texts.iloc[sel["va_pos"]], vec_sel,
        dense_extra=_take_rows(ctx_df, sel["va_pos"]),
    )
    fitted_sel = fit_challenger_candidates(
        Xtr_sel, sel["ytr"], sample_weight=sel["sample_weight"]
    )
    va_pr = {
        name: float(average_precision_score(sel["yva"], clf.predict_proba(Xva_sel)[:, 1]))
        for name, clf in fitted_sel.items()
    }
    best_name = max(va_pr, key=va_pr.get)
    log(
        "[m3] selection validation PR-AUC: "
        + ", ".join(f"{n}={v:.4f}" for n, v in sorted(va_pr.items()))
        + f" -> best={best_name}"
    )

    reports: list[ModelReport] = []
    pos_rate_sel = float(sel["ytr"].mean())
    reports.append(
        ModelReport(
            name="majority_baseline",
            precision=pos_rate_sel,
            recall=1.0,
            f1=2 * pos_rate_sel / (1 + pos_rate_sel) if pos_rate_sel else 0.0,
            pr_auc=pos_rate_sel,
            accuracy=float(1 - pos_rate_sel),
            extras={"note": "always predicts the majority class (selection-train rate)"},
        )
    )
    for name, clf in fitted_sel.items():
        reports.append(
            _metrics(name, sel["yva"], clf.predict_proba(Xva_sel)[:, 1],
                     extras={"scored_on": "purged validation split"})
        )
    if "lightgbm_balanced" not in fitted_sel:
        reports.append(
            ModelReport(
                name="lightgbm_balanced",
                precision=float("nan"), recall=float("nan"), f1=float("nan"),
                pr_auc=float("nan"), accuracy=float("nan"),
                extras={"skipped": "lightgbm unavailable or failed to fit"},
            )
        )
    # Threshold-tuned report on the balanced model (tune on train, eval on
    # validation), mirroring the legacy train() report set.
    thr = tune_threshold(sel["ytr"], fitted_sel["logreg_balanced"].predict_proba(Xtr_sel)[:, 1])
    reports.append(
        _metrics(
            "logreg_balanced_tuned",
            sel["yva"],
            fitted_sel["logreg_balanced"].predict_proba(Xva_sel)[:, 1],
            threshold=thr,
            extras={"threshold_tuned_on": "selection-train", "scored_on": "purged validation split"},
        )
    )
    table = report_table({"reports": reports})
    log(table.to_string(index=False))

    # 2. FINAL TRAIN on all dev, purged vs the untouched test period.
    t0d = pd.to_datetime(df["t0"]).dt.date.to_numpy()
    dev_pos = np.flatnonzero(
        np.array([d <= periods.DEV_END for d in t0d], dtype=bool)
    )
    dev = df.iloc[dev_pos]
    keep = vsplits.purge_against(
        dev["t0"], dev["t1"], periods.TEST_START, periods.TEST_END
    )
    n_purged_final = int((~keep).sum())
    log(
        f"[m3] final train: {int(keep.sum())} dev rows "
        f"(purged {n_purged_final} vs test {periods.TEST_START}..{periods.TEST_END})"
    )
    final = _prepare_fit_frame(
        df, dev_pos[keep.to_numpy()], None, label_cfg, train_label_col, log,
        "final-train",
    )
    vec = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True
    )
    vec.fit(texts.iloc[final["tri_eff"]])
    Xtr = featurize(
        texts.iloc[final["tri_eff"]], vec,
        dense_extra=_take_rows(ctx_df, final["tri_eff"]),
    )
    fitted_final = fit_challenger_candidates(
        Xtr, final["ytr"], sample_weight=final["sample_weight"]
    )
    if best_name not in fitted_final:
        raise RuntimeError(
            f"[m3] selection winner {best_name!r} failed to fit on the final "
            "train block; refusing to persist a different model silently"
        )
    model = fitted_final[best_name]
    joblib.dump(vec, ART / "vectorizer.pkl")
    joblib.dump(model, ART / "model.pkl")
    (ART / "best_model.txt").write_text(best_name)
    log(f"[m3] saved best model: {best_name} (selected on purged validation)")

    label_block = {
        "scheme": label_cfg.scheme,
        "window_days": label_cfg.window_days,
        "quantile": float(label_cfg.quantile),
        "benchmark": label_cfg.benchmark,
        "attention": {
            "enabled": bool(label_cfg.attention_enabled),
            "min_articles": label_cfg.attention_min_articles,
            "top_quartile": label_cfg.attention_top_quartile,
            "quantiles_from": "train only, per fold/block (0.6.0; was global)",
        },
        "cutoff": round(final["cutoff"], 6),
        "cutoff_source": "final-train abn_ret only (0.6.0; was global full-sample)",
        "positive_rate": round(float(final["ytr"].mean()), 4),
        "n_final_train": len(final["tri_eff"]),
        "n_purged_vs_test": n_purged_final,
        "t0_anchor": "first tradable time after publication (0.6.0; was pub calendar date)",
        "periods": periods.describe(),
    }

    # 3. OUT-OF-SAMPLE scores via purged walk-forward on dev (audit 4f).
    from signal_lab.models.walk_forward import run_walk_forward

    wf = run_walk_forward(
        df,
        dense_extra=ctx_df,
        n_splits=5,
        trading_days=trading_days,
        label_cfg=label_cfg,
        model_name=best_name,
        dedupe_policy="weight",
        log=log,
        train_label_col=train_label_col,
        train_label_source=train_labels,
        return_oos=True,
    )
    oos = wf.get("oos_proba", {})
    scored_df = df[["url", "text", "ticker", "published_at", "pub_date"]].copy()
    scored_df["proba"] = np.nan
    if oos:
        idx = np.array(sorted(oos), dtype=int)
        # oos positions are into run_walk_forward's reset work frame, which
        # matches df's order (build_dataset output has no NaN t0/t1/abn_ret).
        assert idx.max() < len(scored_df), "[m3] OOS position out of range"
        scored_df["proba"].iloc[idx] = [oos[i] for i in idx]
    scored_df.to_csv(ART / "scored_news.csv", index=False)
    log(
        f"[m3] scored {len(scored_df)} articles -> {ART / 'scored_news.csv'} "
        f"({len(oos)} with OOS proba; NaN = never in a fold-test block)"
    )

    result = {
        "reports": reports,
        "models": {best_name: model},
        "vectorizer": vec,
        "positive_rate": round(float(final["ytr"].mean()), 4),
        "n_train": len(final["tri_eff"]),
        "n_test": len(oos),
        "best_model": best_name,
        "selection": {
            "valid_start": sp["valid_start"],
            "valid_end": sp["valid_end"],
            "validation_pr_auc": {n: round(v, 4) for n, v in va_pr.items()},
        },
        "walk_forward": wf,
        "label_config": label_block,
        "train_labels": train_labels_block,
    }
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
        featurize(texts, vec, dense_extra=ctx_df),
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
