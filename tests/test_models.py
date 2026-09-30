"""End-to-end training-pipeline test on a tiny synthetic fixture (no network)."""

import pandas as pd

from signal_lab.models import report_table, train


def _fixture(n=120):
    rows = []
    for i in range(n):
        good = i % 3 == 0
        rows.append({
            "title": "record earnings beat, strong revenue growth" if good
                     else "company announces routine board meeting",
            "body_snippet": "",
            "label": 1 if good else 0,
            "published_at": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
        })
    return pd.DataFrame(rows)


def test_train_end_to_end():
    df = _fixture()
    result = train(df, text_col="title")
    assert result["n_train"] > 0 and result["n_test"] > 0
    table = report_table(result)
    assert set(table["model"]) >= {"majority_baseline", "logreg_plain",
                                   "logreg_balanced", "logreg_oversampled",
                                   "logreg_balanced_tuned"}
    # separable fixture: a real model must beat the majority baseline on PR-AUC
    base = table.loc[table.model == "majority_baseline", "pr_auc"].iloc[0]
    best = table.loc[table.model != "majority_baseline", "pr_auc"].max()
    assert best > base, f"model PR-AUC {best} did not beat baseline {base}"
    # predict() works with the fitted vectorizer
    from signal_lab.models import predict

    probs = predict(result["models"]["logreg_balanced"],
                    ["record earnings beat"], vectorizer=result["vectorizer"])
    assert len(probs) == 1 and 0.0 <= probs[0] <= 1.0


def test_lexicon_features():
    from signal_lab.models.lexicon import (LEXICON_FEATURE_NAMES,
                                           lexicon, lexicon_feature_matrix)
    lex = lexicon()
    # authoritative LM dictionary scale: ~2.4k negative, ~350 positive
    assert len(lex["negative"]) > 2000
    assert len(lex["positive"]) > 300
    X = lexicon_feature_matrix([
        "the company reported a loss and faces litigation risk",
        "record earnings beat, strong revenue growth",
        "",
    ])
    assert X.shape == (3, len(LEXICON_FEATURE_NAMES)) == (3, 8)
    # negative-leaning text: negative count > positive count
    assert X[0, 1] > X[0, 0]
    # positive-leaning text: positive count > negative count
    assert X[1, 0] > X[1, 1]
    # empty text: all zeros, no NaN
    assert (X[2] == 0).all()


def test_lexicon_features_flow_into_train():
    df = _fixture()
    result = train(df, text_col="title")
    vec = result["vectorizer"]
    from signal_lab.models import featurize
    X = featurize(["record earnings beat"], vec)
    # TF-IDF vocab width + 8 lexicon columns
    assert X.shape[1] == len(vec.vocabulary_) + 8
