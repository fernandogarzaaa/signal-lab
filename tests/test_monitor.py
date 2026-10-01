"""Tests for workstream 3: the GDELT watchlist monitor.

All network and model I/O is mocked or redirected to tmp dirs; these tests
never touch the real DuckDB, the real GDELT API, or the trained artifacts.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from signal_lab import monitor


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    """Redirect the monitor's state files into a tmp dir."""
    art = tmp_path / "artifacts"
    art.mkdir()
    monkeypatch.setattr(monitor, "ART", art)
    monkeypatch.setattr(monitor, "WATCHLIST_PATH", tmp_path / "watchlist.json")
    monkeypatch.setattr(monitor, "MONITOR_STATE_PATH", art / "monitor_state.json")
    monkeypatch.setattr(monitor, "MONITOR_SCORED_PATH", art / "monitor_scored.csv")
    monkeypatch.setattr(monitor, "SCORED_NEWS_PATH", art / "scored_news.csv")
    return tmp_path


# ---------------------------------------------------------------------------
# Watchlist management
# ---------------------------------------------------------------------------

def test_watchlist_defaults_to_universe(tmp_store):
    tickers = monitor.get_watchlist()
    assert tickers == list(monitor.UNIVERSE)
    assert tmp_store.joinpath("watchlist.json").exists()


def test_watchlist_add_remove_roundtrip(tmp_store):
    assert "NVDA" in monitor.get_watchlist()
    monitor.remove_ticker("NVDA")
    assert "NVDA" not in monitor.get_watchlist()
    monitor.add_ticker("nvda")  # case-insensitive, no duplicates
    tickers = monitor.add_ticker("NVDA")
    assert tickers.count("NVDA") == 1


def test_watchlist_rejects_bad_tickers(tmp_store):
    monitor.get_watchlist()
    with pytest.raises(ValueError):
        monitor.add_ticker("not a ticker!")
    with pytest.raises(ValueError):
        monitor.set_watchlist([])
    with pytest.raises(ValueError):
        monitor.remove_ticker("ZZZZ")


def test_watchlist_corrupt_file_fails_loud(tmp_store):
    tmp_store.joinpath("watchlist.json").write_text("{not json")
    with pytest.raises(ValueError, match="corrupt"):
        monitor.get_watchlist()


# ---------------------------------------------------------------------------
# Loud scorer failures (no silent zero-fill)
# ---------------------------------------------------------------------------

def test_load_scorer_missing_artifacts_raises(tmp_store):
    with pytest.raises(FileNotFoundError, match="missing training artifacts"):
        monitor.load_scorer()


# ---------------------------------------------------------------------------
# Ticker attribution
# ---------------------------------------------------------------------------

def test_attribute_ticker_known_alias():
    ticker, method = monitor._attribute_ticker(
        "Microsoft stock falls as Azure growth slows", "MSFT")
    assert ticker == "MSFT" and method == "alias-extract"


def test_attribute_ticker_unknown_falls_back_to_query():
    ticker, method = monitor._attribute_ticker(
        "Some random text with no company mention", "PLTR")
    assert ticker == "PLTR" and method == "query-fallback"


# ---------------------------------------------------------------------------
# Score merging
# ---------------------------------------------------------------------------

def _scored_frame():
    return pd.DataFrame([
        {"url": "u1", "text": "t1", "ticker": "MSFT",
         "published_at": pd.Timestamp("2026-09-01"), "proba": 0.8},
        {"url": "u2", "text": "t2", "ticker": "MSFT",
         "published_at": pd.Timestamp("2026-09-02"), "proba": 0.2},
    ])


def test_merge_scored_monitor_wins_on_conflict(tmp_store):
    monitor.MONITOR_SCORED_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"url": "u2", "text": "t2-new", "ticker": "MSFT",
         "published_at": "2026-09-03", "pub_date": "2026-09-03",
         "proba": 0.9, "score_note": ""},
        {"url": "u3", "text": "t3", "ticker": "NVDA",
         "published_at": "2026-09-03", "pub_date": "2026-09-03",
         "proba": float("nan"), "score_note": "unscored: test"},
    ]).to_csv(monitor.MONITOR_SCORED_PATH, index=False)
    merged = monitor.merge_scored(_scored_frame())
    assert set(merged["url"]) == {"u1", "u2", "u3"} - {"u3"} | {"u1", "u2"}
    # u2 keeps the monitor's fresher score; unscored u3 is dropped
    assert merged.set_index("url").loc["u2", "proba"] == pytest.approx(0.9)


def test_merge_scored_no_monitor_file_is_identity(tmp_store):
    merged = monitor.merge_scored(_scored_frame())
    assert len(merged) == 2


def test_append_monitor_scored_dedupes_by_url(tmp_store):
    df = pd.DataFrame([
        {"url": "u1", "text": "t1", "ticker": "MSFT",
         "published_at": "2026-09-01", "pub_date": "2026-09-01",
         "proba": 0.7, "score_note": ""},
    ])
    assert monitor.append_monitor_scored(df) == 1
    assert monitor.append_monitor_scored(df) == 0  # duplicate URL ignored
    assert monitor.append_monitor_scored(pd.DataFrame()) == 0


# ---------------------------------------------------------------------------
# Poll cycle (mocked network)
# ---------------------------------------------------------------------------

def _item(url, minutes_ago=5, title="Microsoft stock rises on cloud growth"):
    return {
        "url": url,
        "title": title,
        "body_snippet": "snippet",
        "published_at": datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        "source_domain": "example.com",
        "query": "q",
        "fetched_at": datetime.now(timezone.utc),
    }


def test_poll_once_records_per_ticker_failure_and_continues(tmp_store, monkeypatch):
    calls = []

    def fake_fetch_news(query, start, end, max_records=250, fetch_bodies=True):
        calls.append(query)
        if "Microsoft" in query:
            raise RuntimeError("GDELT exploded")
        return [_item("http://x/nvda1")]

    def fake_prices_for(tickers, days=120):
        return pd.DataFrame(), 0

    def fake_score(items, queried, log=print):
        return pd.DataFrame([
            {"url": it["url"], "text": it["title"], "ticker": queried,
             "published_at": it["published_at"], "pub_date": "2026-10-01",
             "proba": 0.75, "score_note": ""}
            for it in items
        ])

    monkeypatch.setattr(monitor, "fetch_news", fake_fetch_news)
    monkeypatch.setattr(monitor, "_prices_for", fake_prices_for)
    monkeypatch.setattr(monitor, "store_news", lambda items: len(items))
    monkeypatch.setattr(monitor, "score_new_articles", fake_score)
    monkeypatch.setattr(monitor, "POLITENESS_SLEEP_S", 0)

    state = monitor.poll_once(["MSFT", "NVDA"], lookback_min=30, sleep_s=0)

    assert state["status"] == "degraded"
    assert state["tickers"]["MSFT"]["status"] == "failed"
    assert "GDELT exploded" in state["tickers"]["MSFT"]["error"]
    assert state["tickers"]["NVDA"]["status"] == "ok"
    assert state["tickers"]["NVDA"]["articles_new"] == 1
    assert state["totals"]["articles_scored"] == 1
    # state persisted for the dashboard
    persisted = monitor.read_state()
    assert persisted["tickers"]["MSFT"]["status"] == "failed"
    assert "GDELT DOC API" in persisted["gdelt_note"]


def test_poll_once_filters_to_lookback_window(tmp_store, monkeypatch):
    seen = []

    def fake_fetch_news(query, start, end, max_records=250, fetch_bodies=True):
        return [_item("http://x/recent", minutes_ago=5),
                _item("http://x/old", minutes_ago=500)]

    monkeypatch.setattr(monitor, "fetch_news", fake_fetch_news)
    monkeypatch.setattr(monitor, "_prices_for", lambda t, days=120: (pd.DataFrame(), 0))
    monkeypatch.setattr(monitor, "store_news", lambda items: seen.extend(items) or len(items))
    empty_scored = pd.DataFrame(
        columns=["url", "text", "ticker", "published_at", "pub_date",
                 "proba", "score_note"])
    monkeypatch.setattr(monitor, "score_new_articles",
                        lambda items, q, log=print: empty_scored)
    state = monitor.poll_once(["MSFT"], lookback_min=30, sleep_s=0)
    assert state["status"] == "ok"
    assert state["tickers"]["MSFT"]["articles_seen"] == 1
    assert seen[0]["url"] == "http://x/recent"


def test_poll_once_price_failure_does_not_kill_news(tmp_store, monkeypatch):
    def boom(tickers, days=120):
        raise RuntimeError("yfinance down")

    monkeypatch.setattr(monitor, "_prices_for", boom)
    monkeypatch.setattr(monitor, "fetch_news",
                        lambda *a, **k: [_item("http://x/1")])
    monkeypatch.setattr(monitor, "store_news", lambda items: len(items))
    empty_scored = pd.DataFrame(
        columns=["url", "text", "ticker", "published_at", "pub_date",
                 "proba", "score_note"])
    monkeypatch.setattr(monitor, "score_new_articles",
                        lambda items, q, log=print: empty_scored)
    state = monitor.poll_once(["MSFT"], sleep_s=0)
    assert state["price_error"] is not None
    assert "yfinance down" in state["price_error"]
    assert state["tickers"]["MSFT"]["status"] == "ok"


def test_read_state_never_polled(tmp_store):
    assert monitor.read_state()["status"] == "never-polled"


# ---------------------------------------------------------------------------
# Dashboard snapshot
# ---------------------------------------------------------------------------

def test_watchlist_snapshot_empty_state(tmp_store, monkeypatch):
    monkeypatch.setattr(monitor, "_read_prices", lambda tickers: pd.DataFrame())
    snap = monitor.watchlist_snapshot()
    assert snap["tickers"] == list(monitor.UNIVERSE)
    assert snap["monitor"]["status"] == "never-polled"
    assert snap["prices"] == {}


# ---------------------------------------------------------------------------
# Feature-space adaptation (workstream 4): the scorer must build exactly
# the feature matrix the loaded model was trained on.
# ---------------------------------------------------------------------------

def _fake_scorer(n_features, tmp_path):
    """A tiny fake (vectorizer, model) pair with a controlled feature count."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    vec = TfidfVectorizer(max_features=10)
    vec.fit(["apple earnings beat", "market flat today"])
    n_tfidf = len(vec.vocabulary_)

    class FakeModel:
        n_features_in_ = n_features

        def predict_proba(self, X):
            assert X.shape[1] == n_features, (
                f"feature mismatch: built {X.shape[1]}, model expects {n_features}"
            )
            return np.tile([0.4, 0.6], (X.shape[0], 1))

    return vec, FakeModel(), "fake"


def test_score_new_articles_adapts_to_pre_ctx_model(tmp_store, monkeypatch):
    """A model trained without context features (5008-dim space) is scored
    on TF-IDF+lexicon only: no 5015-dim crash."""
    from signal_lab.models.lexicon import LEXICON_FEATURE_NAMES

    vec, model, name = _fake_scorer(0, tmp_store)  # placeholder
    n_base = len(vec.vocabulary_) + len(LEXICON_FEATURE_NAMES)
    vec, model, name = _fake_scorer(n_base, tmp_store)
    monkeypatch.setattr(monitor, "load_scorer", lambda: (vec, model, name))
    monkeypatch.setattr(monitor, "_read_prices", lambda tickers: pd.DataFrame())
    monkeypatch.setattr(
        monitor, "build_context_features",
        lambda df, prices, benchmark=None, log=print: pd.DataFrame(
            {c: [float("nan")] for c in monitor.CONTEXT_FEATURE_NAMES},
            index=df.index,
        ),
    )
    items = [{
        "title": "Apple earnings beat", "body_snippet": "",
        "url": "http://x/1", "published_at": datetime.now(timezone.utc),
    }]
    # NaN ctx -> unscored path; use valid ctx to reach the scorer.
    monkeypatch.setattr(
        monitor, "build_context_features",
        lambda df, prices, benchmark=None, log=print: pd.DataFrame(
            {c: [0.0] for c in monitor.CONTEXT_FEATURE_NAMES},
            index=df.index,
        ),
    )
    out = monitor.score_new_articles(items, "AAPL", log=lambda *a, **k: None)
    assert out["proba"].notna().all()
    assert "without context features" in out["score_note"].iloc[0]


def test_score_new_articles_rejects_unknown_feature_space(tmp_store, monkeypatch):
    """A model expecting a feature count that is neither the base space
    nor base+ctx fails loudly instead of scoring garbage."""
    vec, model, name = _fake_scorer(12345, tmp_store)
    monkeypatch.setattr(monitor, "load_scorer", lambda: (vec, model, name))
    monkeypatch.setattr(monitor, "_read_prices", lambda tickers: pd.DataFrame())
    monkeypatch.setattr(
        monitor, "build_context_features",
        lambda df, prices, benchmark=None, log=print: pd.DataFrame(
            {c: [0.0] for c in monitor.CONTEXT_FEATURE_NAMES},
            index=range(len(df)),
        ),
    )
    items = [{
        "title": "Apple earnings beat", "body_snippet": "",
        "url": "http://x/1", "published_at": datetime.now(timezone.utc),
    }]
    with pytest.raises(ValueError, match="unknown feature space"):
        monitor.score_new_articles(items, "AAPL", log=lambda *a, **k: None)
