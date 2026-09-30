"""Tests for 0.3.0 workstream 1: multi-day labeling, attention filter,
label_config block, and the gold-set sampler/agreement report."""

import pandas as pd
import pytest

from signal_lab.models.gold_set import (
    agreement_report,
    cohens_kappa,
    sample_gold_set,
)
from signal_lab.models.labels import (
    LabelConfig,
    apply_attention,
    baseline_config,
    build_labels,
    forward_cumulative_abnormal,
    label_config_block,
)


def _prices():
    """Two tickers, 10 trading days. AAA drifts flat; BBB jumps +20% on the
    6th trading day (2026-01-08). SPY is flat."""
    dates = pd.date_range("2026-01-01", periods=10, freq="B").date
    rows = []
    for t, base, jump in [
        ("AAA", 100.0, 0.0),
        ("BBB", 50.0, 0.20),
        ("SPY", 400.0, 0.0),
    ]:
        close = base
        for i, d in enumerate(dates):
            if t == "BBB" and i == 5:  # jump into 2026-01-08
                close *= 1.0 + jump
            rows.append({"ticker": t, "date": d, "close": close})
    return pd.DataFrame(rows)


def _news(rows):
    """rows: list of (url, ticker, pub_date_str)."""
    return pd.DataFrame(
        [
            {
                "url": u,
                "published_at": pd.Timestamp(p),
                "pub_date": pd.Timestamp(p).date(),
                "ticker": t,
            }
            for u, t, p in rows
        ]
    )


def test_forward_cumulative_matches_manual_product():
    cfg = LabelConfig(window_days=3)
    fwd = forward_cumulative_abnormal(_prices(), cfg)
    # BBB on 2026-01-06: the +20% jump lands on the 2nd forward day.
    row = fwd[(fwd.ticker == "BBB") & (fwd.date == pd.Timestamp("2026-01-06").date())]
    assert row["fwd_days"].iloc[0] == 3
    assert row["ticker_fwd_ret"].iloc[0] == pytest.approx(0.20, abs=1e-9)
    assert row["abn_ret"].iloc[0] == pytest.approx(0.20, abs=1e-9)  # SPY flat
    # window=1 on the same date sees no jump yet.
    fwd1 = forward_cumulative_abnormal(_prices(), LabelConfig(window_days=1))
    row1 = fwd1[
        (fwd1.ticker == "BBB") & (fwd1.date == pd.Timestamp("2026-01-06").date())
    ]
    assert row1["abn_ret"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_window_days_validation():
    with pytest.raises(ValueError):
        LabelConfig(window_days=0).validated()
    with pytest.raises(ValueError):
        LabelConfig(quantile=1.5).validated()
    with pytest.raises(ValueError):
        LabelConfig(scheme="nope").validated()
    with pytest.raises(ValueError):
        LabelConfig(attention_min_articles=0).validated()


def test_baseline_reproduces_legacy_t1_labeling():
    """baseline_config() must match the 0.2.0 inline formula exactly:
    next-day return via shift(-1), SPY benchmark, top-decile cutoff."""
    prices = _prices()
    news = _news(
        [
            ("u1", "AAA", "2026-01-05"),
            ("u2", "AAA", "2026-01-06"),
            ("u3", "BBB", "2026-01-06"),
            ("u4", "BBB", "2026-01-07"),
            ("u5", "AAA", "2026-01-07"),
            ("u6", "BBB", "2026-01-05"),
        ]
    )
    got, _ = build_labels(news, prices, baseline_config())

    # legacy 0.2.0 computation, written out in full
    p = prices.sort_values(["ticker", "date"]).copy()
    p["next_ret"] = p.groupby("ticker")["close"].shift(-1) / p["close"] - 1
    nxt = p[["ticker", "date", "next_ret"]].dropna()
    spy = nxt[nxt.ticker == "SPY"][["date", "next_ret"]].rename(
        columns={"next_ret": "mkt_ret"}
    )
    stock = nxt[nxt.ticker != "SPY"]
    df = news.merge(
        stock[["ticker", "date", "next_ret"]],
        left_on=["ticker", "pub_date"],
        right_on=["ticker", "date"],
        how="left",
    )
    df = df.merge(spy, left_on="pub_date", right_on="date", how="left")
    df = df.dropna(subset=["next_ret"])
    df["abn_ret"] = df["next_ret"] - df["mkt_ret"]
    cutoff = df["abn_ret"].quantile(0.90)
    legacy = (df.set_index("url")["abn_ret"] >= cutoff).astype(int)

    assert set(got["url"]) == set(legacy.index)
    for _, row in got.iterrows():
        assert row["label"] == legacy[row["url"]], row["url"]


def test_windowed_labels_differ_from_t1_when_news_digests_slowly():
    news = _news(
        [
            ("u1", "BBB", "2026-01-06"),  # +20% lands on forward day 2
            ("u2", "BBB", "2026-01-07"),  # +20% lands on forward day 1
            ("u3", "AAA", "2026-01-06"),  # flat
            ("u4", "AAA", "2026-01-07"),  # flat
        ]
    )
    cfg = LabelConfig(window_days=2, quantile=0.5, attention_enabled=False)
    got, _ = build_labels(news, _prices(), cfg)
    by_url = dict(zip(got["url"], got["label"]))
    # both BBB articles see the +20% inside a 2-day window; AAA sees nothing
    assert by_url["u1"] == 1 and by_url["u2"] == 1
    assert by_url["u3"] == 0 and by_url["u4"] == 0


def test_partial_windows_dropped_and_reported():
    prices = _prices()
    news = _news(
        [
            ("u1", "AAA", "2026-01-13"),  # only 1 forward day left
            ("u2", "AAA", "2026-01-09"),
        ]
    )  # 3 forward days left
    cfg = LabelConfig(window_days=3, quantile=0.5, attention_enabled=False)
    got, block = build_labels(news, prices, cfg)
    assert set(got["url"]) == {"u2"}
    assert block["partial_window_rows_dropped"] == 1


def test_attention_min_articles():
    df = pd.DataFrame(
        [
            {
                "url": "a1",
                "ticker": "AAA",
                "pub_date": pd.Timestamp("2026-01-05").date(),
            },
            {
                "url": "b1",
                "ticker": "BBB",
                "pub_date": pd.Timestamp("2026-01-05").date(),
            },
            {
                "url": "b2",
                "ticker": "BBB",
                "pub_date": pd.Timestamp("2026-01-05").date(),
            },
            {
                "url": "b3",
                "ticker": "BBB",
                "pub_date": pd.Timestamp("2026-01-05").date(),
            },
        ]
    )
    kept, info = apply_attention(df, min_articles=2, top_quartile=False)
    assert set(kept["url"]) == {"b1", "b2", "b3"}
    assert info["dropped_rows"] == 1
    assert info["dropped_pct"] == pytest.approx(0.25)
    assert (kept["n_articles"] == 3).all()


def test_attention_top_quartile_branch():
    # AAA has quiet days (1 article) and one busy day (12 articles).
    rows = [
        {
            "url": f"q{i}",
            "ticker": "AAA",
            "pub_date": pd.Timestamp(f"2026-01-{5 + i:02d}").date(),
        }
        for i in range(4)
    ]
    rows += [
        {
            "url": f"busy{i}",
            "ticker": "AAA",
            "pub_date": pd.Timestamp("2026-01-20").date(),
        }
        for i in range(12)
    ]
    df = pd.DataFrame(rows)
    kept, _ = apply_attention(df, min_articles=5, top_quartile=True)
    assert set(kept["url"]) == {f"busy{i}" for i in range(12)}
    kept2, info2 = apply_attention(df, min_articles=5, top_quartile=False)
    assert len(kept2) == 12  # only the busy day passes the raw count rule
    assert info2["dropped_rows"] == 4


def test_label_config_block_shape():
    cfg = LabelConfig(window_days=3, quantile=0.9, attention_min_articles=2)
    news = _news(
        [
            ("u1", "AAA", "2026-01-05"),
            ("u2", "AAA", "2026-01-05"),
            ("u3", "BBB", "2026-01-06"),
            ("u4", "BBB", "2026-01-06"),
        ]
    )
    got, block = build_labels(news, _prices(), cfg)
    assert set(block) == {
        "scheme",
        "window_days",
        "quantile",
        "benchmark",
        "attention",
        "cutoff",
        "positive_rate",
        "n_labeled",
        "partial_window_rows_dropped",
    }
    assert set(block["attention"]) == {
        "enabled",
        "min_articles",
        "top_quartile",
        "dropped_rows",
        "dropped_pct",
    }
    assert block["scheme"] == "windowed" and block["window_days"] == 3
    assert block["n_labeled"] == len(got)
    assert block["positive_rate"] == pytest.approx(got["label"].mean())
    assert isinstance(block["cutoff"], float)
    # JSON-serializable with only plain Python types
    import json

    json.dumps(block)


def test_build_labels_column_contract():
    """build_labels returns the documented column contract (the DB-touching
    build_dataset wrapper is exercised in the integration check, not here)."""
    cfg = LabelConfig(window_days=2, quantile=0.5, attention_enabled=False)
    news = _news([("u1", "BBB", "2026-01-06"), ("u2", "AAA", "2026-01-06")])
    got, block = build_labels(news, _prices(), cfg)
    assert list(got.columns) == [
        "url",
        "ticker",
        "published_at",
        "pub_date",
        "fwd_days",
        "n_articles",
        "ticker_fwd_ret",
        "mkt_fwd_ret",
        "abn_ret",
        "label",
    ]
    assert block["n_labeled"] == 2


# --- gold set -------------------------------------------------------------


def _labeled_pool():
    return pd.DataFrame(
        {
            "url": [f"u{i}" for i in range(20)],
            "label": [1] * 6 + [0] * 14,
        }
    )


def test_sample_gold_set_stratified_and_deterministic():
    df = _labeled_pool()
    a = sample_gold_set(df, n=10, seed=7)
    b = sample_gold_set(df, n=10, seed=7)
    assert a["url"].tolist() == b["url"].tolist()  # deterministic
    assert len(a) == 10 and a["url"].is_unique
    assert a["label"].sum() == 5  # pos_frac=0.5 of 10
    # short class refills: only 2 positives available
    small = pd.DataFrame(
        {"url": ["p1", "p2", "n1", "n2", "n3"], "label": [1, 1, 0, 0, 0]}
    )
    s = sample_gold_set(small, n=5, seed=1)
    assert len(s) == 5 and s["label"].sum() == 2


def test_sample_gold_set_validation():
    with pytest.raises(ValueError):
        sample_gold_set(_labeled_pool(), n=0, seed=1)


def test_cohens_kappa_known_values():
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    # 3/4 agree; pe = 0.5*0.5 + 0.5*0.5 = 0.5 -> kappa = (0.75-0.5)/0.5 = 0.5
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)


def test_agreement_report_known_values():
    items = [
        {"weak_label": 1, "human_label": 1},
        {"weak_label": 1, "human_label": 0},
        {"weak_label": 0, "human_label": 0},
        {"weak_label": 0, "human_label": 0},
        {"weak_label": 1, "human_label": None},  # skipped: excluded
    ]
    rep = agreement_report(items)
    assert rep["n_items"] == 5 and rep["n_judged"] == 4
    assert rep["agreement"] == pytest.approx(0.75)
    assert (rep["tp"], rep["tn"], rep["fp"], rep["fn"]) == (1, 2, 1, 0)
    assert rep["cohens_kappa"] == pytest.approx(0.5)
    assert rep["weak_positive_rate"] == pytest.approx(0.5)
    assert rep["human_positive_rate"] == pytest.approx(0.25)


def test_agreement_report_no_judgments():
    rep = agreement_report([{"weak_label": 1, "human_label": None}])
    assert rep["n_judged"] == 0 and rep["agreement"] is None
    assert rep["cohens_kappa"] is None


def test_label_config_block_builder():
    df = pd.DataFrame({"label": [1, 0, 0, 0]})
    cfg = LabelConfig()
    block = label_config_block(
        df,
        cfg,
        cutoff=0.05,
        attention_info={
            "dropped_rows": 2,
            "dropped_pct": 0.33,
            "min_articles": 2,
            "top_quartile": True,
        },
        partial_dropped=1,
    )
    assert block["attention"]["enabled"] is True
    assert block["attention"]["dropped_rows"] == 2
    assert block["positive_rate"] == pytest.approx(0.25)
    assert block["partial_window_rows_dropped"] == 1
    # attention disabled -> zeroed stats, still present
    block2 = label_config_block(
        df, baseline_config(), cutoff=0.05, attention_info=None, partial_dropped=0
    )
    assert block2["attention"]["enabled"] is False
    assert block2["attention"]["dropped_rows"] == 0
