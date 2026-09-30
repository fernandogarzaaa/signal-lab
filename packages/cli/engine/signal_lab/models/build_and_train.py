"""Build the M3 weak-label dataset and train the sentiment classifier.

Dataset: each news article -> (text, ticker, published_date, label) where
label = 1 if the ticker's next-trading-day abnormal return (vs SPY) lands in
the top decile, else 0. See models.__doc__ for the documented labeling bias.

Saves:
  data/artifacts/vectorizer.pkl, model.pkl (joblib)
  data/artifacts/scored_news.csv (url, ticker, published_at, proba for M4/M6)

Usage: python -m signal_lab.models.build_and_train
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from signal_lab.ingest import DB_PATH, fetch_prices, store_prices
from signal_lab.models import make_text, report_table, train
from signal_lab.nlp import extract_baseline

REPO_ROOT = Path(__file__).resolve().parents[3]
ART = REPO_ROOT / "data" / "artifacts"


def _next_day_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, date): next trading day's close-to-close return."""
    p = prices.sort_values(["ticker", "date"]).copy()
    p["next_date"] = p.groupby("ticker")["date"].shift(-1)
    p["next_close"] = p.groupby("ticker")["close"].shift(-1)
    p["next_ret"] = p["next_close"] / p["close"] - 1
    return p[["ticker", "date", "next_ret"]].dropna()


def build_dataset(log=print) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    news = con.execute(
        "SELECT url, title, body_snippet, published_at FROM news_raw WHERE title <> ''"
    ).fetchdf()
    prices = con.execute("SELECT ticker, date, close FROM prices_daily").fetchdf()
    con.close()
    prices["date"] = pd.to_datetime(prices["date"]).dt.date

    # Ensure a market benchmark exists. (Date strings must be plain
    # YYYY-MM-DD: yfinance rejects datetime strings with a time component.)
    if "SPY" not in set(prices["ticker"]):
        log("[m3] fetching SPY benchmark prices")
        spy = fetch_prices(["SPY"], prices["date"].min().isoformat(),
                           prices["date"].max().isoformat())
        store_prices(spy)
        con = duckdb.connect(DB_PATH, read_only=True)
        prices = con.execute("SELECT ticker, date, close FROM prices_daily").fetchdf()
        con.close()
        prices["date"] = pd.to_datetime(prices["date"]).dt.date

    news["pub_date"] = pd.to_datetime(news["published_at"], utc=True).dt.date

    # Ticker per article via the deterministic baseline extractor.
    # Match on title + snippet: GDELT often returns the company mention in
    # the body rather than the headline.
    tickers, dropped = [], 0
    news["text"] = (news["title"].fillna("") + " " +
                    news["body_snippet"].fillna("")).str.strip()
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

    nxt = _next_day_returns(prices)
    spy_ret = nxt[nxt.ticker == "SPY"][["date", "next_ret"]].rename(
        columns={"next_ret": "mkt_ret"})
    stock = nxt[nxt.ticker != "SPY"]
    df = news.merge(stock[["ticker", "date", "next_ret"]],
                    left_on=["ticker", "pub_date"], right_on=["ticker", "date"],
                    how="left")
    df = df.merge(spy_ret, left_on="pub_date", right_on="date", how="left",
                  suffixes=("", "_mkt"))
    df = df.dropna(subset=["next_ret"]).reset_index(drop=True)
    # Fallback: universe equal-weight mean when SPY is missing that day.
    if df["mkt_ret"].isna().any():
        umean = stock.groupby("date")["next_ret"].mean().rename("umean")
        df = df.merge(umean, left_on="pub_date", right_index=True, how="left")
        df["mkt_ret"] = df["mkt_ret"].fillna(df["umean"])
    df["abn_ret"] = df["next_ret"] - df["mkt_ret"]
    if df.empty:
        raise ValueError(
            "[m3] no labeled rows: none of the news articles could be matched "
            "to a next-trading-day price (articles may be newer than the "
            "latest prices, or entity extraction found no tickers). "
            "Ingest an older news window before training.")
    cutoff = df["abn_ret"].quantile(0.90)
    df["label"] = (df["abn_ret"] >= cutoff).astype(int)
    log(f"[m3] labeled rows: {len(df)}, positive rate: {df['label'].mean():.3f}, "
          f"top-decile cutoff: {cutoff:+.4f}")
    return df[["url", "text", "ticker", "published_at", "pub_date",
               "next_ret", "mkt_ret", "abn_ret", "label"]]


def run_pipeline(log=print) -> dict:
    """Build dataset, train, persist artifacts. Returns the train() result."""
    ART.mkdir(parents=True, exist_ok=True)
    df = build_dataset(log=log)
    result = train(df, text_col="text")
    table = report_table(result)
    log(table.to_string(index=False))
    log(f"n_train={result['n_train']} n_test={result['n_test']} "
        f"positive_rate={result['positive_rate']:.3f}")

    # Persist the best model (highest test PR-AUC among fitted models).
    scored = [(r.pr_auc, r.name) for r in result["reports"] if r.name in result["models"]]
    best_name = max(scored)[1]
    joblib.dump(result["vectorizer"], ART / "vectorizer.pkl")
    joblib.dump(result["models"][best_name], ART / "model.pkl")
    (ART / "best_model.txt").write_text(best_name)
    log(f"[m3] saved best model: {best_name}")

    # Score every article for M4/M6.
    vec = result["vectorizer"]
    model = result["models"][best_name]
    proba = model.predict_proba(vec.transform(make_text(df)))[:, 1]
    scored_df = df[["url", "text", "ticker", "published_at", "pub_date"]].copy()
    scored_df["proba"] = proba
    scored_df.to_csv(ART / "scored_news.csv", index=False)
    log(f"[m3] scored {len(scored_df)} articles -> {ART / 'scored_news.csv'}")
    result["best_model"] = best_name
    return result


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
