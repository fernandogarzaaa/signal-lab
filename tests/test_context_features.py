"""Tests for 0.3.0 workstream 3 market-context features.

Every price feature is verified against hand-computed values on synthetic
series, and the no-leakage test pins the core guarantee: features for an
article may only use price data from trading days strictly before the
article's publish date.
"""

import numpy as np
import pandas as pd
import pytest

from signal_lab.models.context_features import (
    CONTEXT_FEATURE_NAMES,
    build_context_features,
    source_domain_tier,
)


def _prices(ticker, closes, start="2026-01-05"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": [d.date() for d in dates],
            "close": [float(c) for c in closes],
        }
    )


def _news_row(url, ticker, pub_date, text="some market news text here"):
    return {
        "url": url,
        "ticker": ticker,
        "published_at": pd.Timestamp(pub_date, tz="UTC"),
        "pub_date": pub_date,
        "text": text,
    }


def _linear_closes(n, start=100.0, step=1.0):
    return [start + step * i for i in range(n)]


def test_feature_names_contract():
    assert CONTEXT_FEATURE_NAMES == [
        "ctx_vol20_pct",
        "ctx_mom5",
        "ctx_mom20",
        "ctx_sector_rel5",
        "ctx_src_tier",
        "ctx_word_count",
        "ctx_hour_bucket",
    ]


def test_momentum_hand_computed():
    # 30 trading days, close = 100 + i. Article published on the calendar
    # date of trading day 29 (i.e. intraday on the last day): as-of trading
    # day is 28, so mom5 = close28/close23 - 1 = 128/123 - 1 and
    # mom20 = close28/close8 - 1 = 128/108 - 1.
    closes = _linear_closes(30)
    prices = _prices("AAA", closes)
    asof_day = pd.bdate_range("2026-01-05", periods=30)[29].date()
    news = pd.DataFrame([_news_row("https://example.com/a", "AAA", asof_day)])
    out = build_context_features(news, prices, log=lambda *a, **k: None)
    assert out["ctx_mom5"].iloc[0] == pytest.approx(128.0 / 123.0 - 1.0)
    assert out["ctx_mom20"].iloc[0] == pytest.approx(128.0 / 108.0 - 1.0)


def test_volatility_percentile_constant_series():
    # Constant price -> every 20d vol is exactly 0, so the current vol ranks
    # at the very top of its own history: percentile 1.0.
    prices = _prices("AAA", [100.0] * 60)
    pub = pd.bdate_range("2026-01-05", periods=61)[60].date()
    news = pd.DataFrame([_news_row("https://example.com/a", "AAA", pub)])
    out = build_context_features(news, prices, log=lambda *a, **k: None)
    assert out["ctx_vol20_pct"].iloc[0] == pytest.approx(1.0)


def test_volatility_percentile_calm_after_storm():
    # 40 days alternating 100/110 (high vol), then 20 days flat at 110.
    # As of the last day the trailing 20d vol is exactly 0 while every one
    # of the 40 prior 20d vols is positive -> percentile 0.0.
    closes = [100.0 if i % 2 == 0 else 110.0 for i in range(40)] + [110.0] * 20
    prices = _prices("AAA", closes)
    pub = pd.bdate_range("2026-01-05", periods=61)[60].date()
    news = pd.DataFrame([_news_row("https://example.com/a", "AAA", pub)])
    out = build_context_features(news, prices, log=lambda *a, **k: None)
    assert out["ctx_vol20_pct"].iloc[0] == pytest.approx(0.0)


def test_volatility_percentile_independent_loop():
    # Same calm-after-storm series, verified by a plain loop written
    # independently of the module's vectorized implementation.
    closes = [100.0 if i % 2 == 0 else 110.0 for i in range(40)] + [110.0] * 20
    prices = _prices("AAA", closes)
    pub = pd.bdate_range("2026-01-05", periods=61)[60].date()
    news = pd.DataFrame([_news_row("https://example.com/a", "AAA", pub)])
    out = build_context_features(news, prices, log=lambda *a, **k: None)

    rets = np.array(closes[1:]) / np.array(closes[:-1]) - 1.0
    # Module day D uses returns D-19..D, i.e. rets[D-20..D-1] here.
    # Trailing 20d vols for module days 20..59 (day 19 and earlier have no
    # full 20-return window).
    vols = [float(np.std(rets[d - 20 : d], ddof=1)) for d in range(20, 60)]
    expected = float(np.mean([v <= vols[-1] for v in vols[:-1]]))
    assert out["ctx_vol20_pct"].iloc[0] == pytest.approx(expected)
    assert expected == pytest.approx(0.0)


def test_sector_relative_strength_spy_fallback():
    # No sector_prices given -> the sector leg falls back to SPY.
    # AAA: close = 100 + i ; SPY: close = 200 + 0.5 i ; 30 trading days,
    # article on the calendar date of trading day 29 (as-of = day 28).
    prices = pd.concat(
        [
            _prices("AAA", _linear_closes(30, 100.0, 1.0)),
            _prices("SPY", _linear_closes(30, 200.0, 0.5)),
        ]
    )
    pub = pd.bdate_range("2026-01-05", periods=30)[29].date()
    news = pd.DataFrame([_news_row("https://example.com/a", "AAA", pub)])
    out = build_context_features(news, prices, log=lambda *a, **k: None)
    mom5_aaa = 128.0 / 123.0 - 1.0
    mom5_spy = (200.0 + 0.5 * 28) / (200.0 + 0.5 * 23) - 1.0
    assert out["ctx_sector_rel5"].iloc[0] == pytest.approx(mom5_aaa - mom5_spy)


def test_sector_relative_strength_real_etf_leg():
    # With sector_prices given, AAPL (mapped to XLK) uses the XLK leg.
    prices = pd.concat(
        [
            _prices("AAPL", _linear_closes(30, 100.0, 1.0)),
            _prices("SPY", _linear_closes(30, 200.0, 0.5)),
        ]
    )
    sector = _prices("XLK", _linear_closes(30, 50.0, 2.0))
    pub = pd.bdate_range("2026-01-05", periods=30)[29].date()
    news = pd.DataFrame([_news_row("https://example.com/a", "AAPL", pub)])
    out = build_context_features(
        news, prices, sector_prices=sector, log=lambda *a, **k: None
    )
    mom5_aapl = 128.0 / 123.0 - 1.0
    mom5_xlk = (50.0 + 2.0 * 28) / (50.0 + 2.0 * 23) - 1.0
    assert out["ctx_sector_rel5"].iloc[0] == pytest.approx(mom5_aapl - mom5_xlk)


def test_source_domain_tier():
    assert source_domain_tier("https://www.reuters.com/world/x") == 2
    assert source_domain_tier("https://bloomberg.com/news/y") == 2
    assert source_domain_tier("https://markets.businessinsider.com/news/z") == 1
    assert source_domain_tier("https://www.nasdaq.com/articles/q") == 1
    assert source_domain_tier("https://example-blog.xyz/x") == 0
    assert source_domain_tier("not a url") == 0
    assert source_domain_tier("") == 0


def test_article_metadata():
    news = pd.DataFrame(
        [
            _news_row(
                "https://www.reuters.com/a",
                "AAA",
                pd.Timestamp("2026-03-04").date(),
                text="Hello world, hello!",
            )
        ]
    )
    news.loc[0, "published_at"] = pd.Timestamp("2026-03-04T14:30:00Z")
    prices = _prices("AAA", [100.0] * 60)
    out = build_context_features(news, prices, log=lambda *a, **k: None)
    assert out["ctx_src_tier"].iloc[0] == 2
    assert out["ctx_word_count"].iloc[0] == 3
    assert out["ctx_hour_bucket"].iloc[0] == 2  # 14:xx UTC -> bucket 2

    news2 = news.copy()
    news2.loc[0, "published_at"] = pd.Timestamp("2026-03-04T03:00:00Z")
    out2 = build_context_features(news2, prices, log=lambda *a, **k: None)
    assert out2["ctx_hour_bucket"].iloc[0] == 0
    news2.loc[0, "published_at"] = pd.Timestamp("2026-03-04T23:59:00Z")
    out3 = build_context_features(news2, prices, log=lambda *a, **k: None)
    assert out3["ctx_hour_bucket"].iloc[0] == 3


def test_insufficient_history_yields_nan():
    # Only 10 trading days: mom20/vol20 cannot be computed -> NaN, so the
    # pipeline can drop and count the row.
    prices = _prices("AAA", _linear_closes(10))
    pub = pd.bdate_range("2026-01-05", periods=11)[10].date()
    news = pd.DataFrame([_news_row("https://example.com/a", "AAA", pub)])
    out = build_context_features(news, prices, log=lambda *a, **k: None)
    assert out["ctx_mom20"].isna().iloc[0]
    assert out["ctx_vol20_pct"].isna().iloc[0]
    # mom5 needs only 5 returns: computable.
    assert not out["ctx_mom5"].isna().iloc[0]


def test_no_leakage_post_article_spike():
    # Article published on the calendar date of trading day 60 (as-of day
    # 59). A 3x price spike on days 70+ must not move any price feature;
    # a spike on day 50 (before the as-of day) must move them, proving the
    # test is not vacuous.
    closes = _linear_closes(80)
    prices = _prices("BBB", closes)
    pub = pd.bdate_range("2026-01-05", periods=80)[60].date()
    news = pd.DataFrame([_news_row("https://example.com/a", "BBB", pub)])
    quiet = {"log": lambda *a, **k: None}
    base = build_context_features(news, prices, **quiet)
    price_cols = ["ctx_vol20_pct", "ctx_mom5", "ctx_mom20", "ctx_sector_rel5"]

    # Sanity: the as-of day really is trading day 59.
    assert base["ctx_mom5"].iloc[0] == pytest.approx(159.0 / 154.0 - 1.0)

    spiked = prices.copy()
    spiked.loc[spiked["date"] >= spiked["date"].iloc[70], "close"] *= 3.0
    after = build_context_features(news, spiked, **quiet)
    pd.testing.assert_frame_equal(
        base[price_cols], after[price_cols], check_exact=False
    )

    early = prices.copy()
    early.loc[early["date"] >= early["date"].iloc[50], "close"] *= 3.0
    moved = build_context_features(news, early, **quiet)
    assert not moved[price_cols].equals(base[price_cols])


def test_dense_extra_flows_through_train_and_predict():
    from signal_lab.models import feature_names, featurize, predict, train

    rows = []
    for i in range(120):
        good = i % 3 == 0
        rows.append(
            {
                "title": "record earnings beat, strong revenue growth"
                if good
                else "company announces routine board meeting",
                "body_snippet": "",
                "label": 1 if good else 0,
                "published_at": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            }
        )
    df = pd.DataFrame(rows)
    rng = np.random.RandomState(0)
    dense = pd.DataFrame(
        rng.normal(size=(len(df), len(CONTEXT_FEATURE_NAMES))),
        columns=CONTEXT_FEATURE_NAMES,
    )
    result = train(df, text_col="title", dense_extra=dense)
    vec = result["vectorizer"]
    names = feature_names(vec, CONTEXT_FEATURE_NAMES)
    X = featurize(["record earnings beat"], vec, dense_extra=dense.iloc[[0]])
    assert X.shape[1] == len(names) == len(vec.vocabulary_) + 8 + 7
    probs = predict(
        result["models"]["logreg_balanced"],
        ["record earnings beat"],
        vectorizer=vec,
        dense_extra=dense.iloc[[0]],
    )
    assert len(probs) == 1 and 0.0 <= probs[0] <= 1.0


def test_dense_extra_nan_fails_loud():
    from sklearn.feature_extraction.text import TfidfVectorizer

    from signal_lab.models import featurize

    vec = TfidfVectorizer().fit(["hello world"])
    bad = pd.DataFrame({"ctx_mom5": [np.nan]})
    with pytest.raises(ValueError, match="NaN"):
        featurize(["hello world"], vec, dense_extra=bad)


def test_feature_importance_block_shape():
    from signal_lab.models import feature_names, train
    from signal_lab.models.build_and_train import feature_importance_block
    from signal_lab.models.lexicon import LEXICON_FEATURE_NAMES

    rows = []
    for i in range(120):
        good = i % 3 == 0
        rows.append(
            {
                "title": "record earnings beat, strong revenue growth"
                if good
                else "company announces routine board meeting",
                "body_snippet": "",
                "label": 1 if good else 0,
                "published_at": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            }
        )
    df = pd.DataFrame(rows)
    rng = np.random.RandomState(1)
    dense = pd.DataFrame(
        rng.normal(size=(len(df), len(CONTEXT_FEATURE_NAMES))),
        columns=CONTEXT_FEATURE_NAMES,
    )
    result = train(df, text_col="title", dense_extra=dense)
    vec = result["vectorizer"]
    names = feature_names(vec, CONTEXT_FEATURE_NAMES)
    dense_names = list(LEXICON_FEATURE_NAMES) + list(CONTEXT_FEATURE_NAMES)
    from signal_lab.models import featurize as _fz
    from signal_lab.models import make_text

    X = _fz(make_text(df), vec, dense_extra=dense)
    block = feature_importance_block(
        "logreg_balanced",
        result["models"]["logreg_balanced"],
        names,
        dense_names,
        X,
    )
    assert block["model"] == "logreg_balanced"
    assert block["method"] == "abs_coef_times_std"
    assert block["n_features"] == len(names) == len(vec.vocabulary_) + 15
    assert len(block["top_text_terms"]) <= 20
    assert len(block["dense_features"]) == 15
    got = {d["feature"] for d in block["dense_features"]}
    assert set(dense_names) == got
    for d in block["dense_features"] + block["top_text_terms"]:
        assert np.isfinite(d["importance"]) and d["importance"] >= 0
    imps = [d["importance"] for d in block["dense_features"]]
    assert imps == sorted(imps, reverse=True)
