"""M3: classical sentiment classifier on imbalanced data.

Weak labels: for each news item, next-day abnormal return of its ticker
(ticker return minus SPY market return). Positive class = top decile of
abnormal returns; everything else = negative class.

Labeling bias (documented, not hidden):
  1. Price move is not sentiment. A price jump can come from market-wide
     moves (mitigated by subtracting the market), scheduled earnings, or
     macro news the article did not cause.
  2. Items sharing a ticker-day share a label, so samples are correlated;
     the train/test split is temporal to avoid leaking same-day information.
  3. Selection bias: only 10 large-cap tickers, English finance-filtered
     GDELT queries. Nothing here generalizes to small caps or other languages.
  4. No lookahead: features are text published at time t; labels use t+1
     returns. The split is by calendar date, never shuffled.

Metrics: precision, recall, F1, PR-AUC. Accuracy is reported only to show
why it lies on imbalanced data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score)
from sklearn.utils import resample

from signal_lab.models.lexicon import (LEXICON_FEATURE_NAMES,
                                       lexicon_feature_matrix)

try:
    from scipy.sparse import csr_matrix, hstack as sparse_hstack
except ImportError:  # pragma: no cover - scipy is a hard dependency
    csr_matrix = None
    sparse_hstack = None

RANDOM_STATE = 7
TEXT_COLS = ["title", "body_snippet"]


def make_text(df: pd.DataFrame) -> pd.Series:
    if "text" in df.columns:
        return df["text"].fillna("").str.strip()
    return (df["title"].fillna("") + " " + df["body_snippet"].fillna("")).str.strip()


def temporal_split(dates: pd.Series, train_frac: float = 0.7):
    """Split indices by calendar order. Returns (train_idx, test_idx)."""
    order = np.argsort(dates.values)
    cut = int(len(order) * train_frac)
    return order[:cut], order[cut:]


def oversample_minority(X, y, random_state: int = RANDOM_STATE):
    """Manual random oversampling of the minority class (no imblearn needed)."""
    X = X.tocsr() if hasattr(X, "tocsr") else X
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    Xs, ys = [X], [y]
    for c in classes:
        n_missing = target - (y == c).sum()
        if n_missing > 0:
            Xi, yi = resample(X[y == c], y[y == c], replace=True,
                              n_samples=n_missing, random_state=random_state)
            Xs.append(Xi)
            ys.append(yi)
    from scipy.sparse import vstack as sp_vstack
    Xb = sp_vstack(Xs) if hasattr(X, "tocsr") else np.vstack(Xs)
    return Xb, np.concatenate(ys)


@dataclass
class ModelReport:
    name: str
    precision: float
    recall: float
    f1: float
    pr_auc: float
    accuracy: float
    threshold: float = 0.5
    extras: dict = field(default_factory=dict)


def _metrics(name, y_true, scores, threshold=0.5, extras=None) -> ModelReport:
    y_pred = (scores >= threshold).astype(int)
    return ModelReport(
        name=name,
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        pr_auc=average_precision_score(y_true, scores),
        accuracy=float((y_pred == y_true).mean()),
        threshold=threshold,
        extras=extras or {},
    )


def tune_threshold(y_true, scores) -> float:
    """Pick the threshold maximizing F1 on the given (validation) labels."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    best = int(np.argmax(f1))
    return float(thr[best]) if best < len(thr) else 0.5


def featurize(texts, vectorizer):
    """Full feature matrix: TF-IDF (sparse) + 8 LM lexicon features (dense).

    Used everywhere a fitted vectorizer transforms raw text (train/test
    split, predict(), artifact scoring) so the model always sees the same
    feature space it was trained on.
    """
    X_tfidf = vectorizer.transform(list(texts))
    X_lex = csr_matrix(lexicon_feature_matrix(texts))
    return sparse_hstack([X_tfidf, X_lex], format="csr")


def train(df: pd.DataFrame, text_col: str = "title"):
    """Fit baseline + challengers. Returns {'models': {...}, 'reports': [...],
    'vectorizer': vec, 'positive_rate': float} on a temporal held-out split.

    df must have columns: published_at (datetime), text_col(s), label (0/1).
    """
    df = df.dropna(subset=[text_col, "label", "published_at"]).reset_index(drop=True)
    if text_col == "title":
        texts = make_text(df)
    else:
        texts = df[text_col].fillna("").astype(str)
    y = df["label"].astype(int).values
    tr_idx, te_idx = temporal_split(pd.to_datetime(df["published_at"]))

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                          stop_words="english", sublinear_tf=True)
    vec.fit(texts.iloc[tr_idx])
    Xtr = featurize(texts.iloc[tr_idx], vec)
    Xte = featurize(texts.iloc[te_idx], vec)
    ytr, yte = y[tr_idx], y[te_idx]
    pos_rate = float(y.mean())

    models: dict = {}
    reports: list[ModelReport] = []

    # 0. Majority-class baseline: PR-AUC of a constant scorer equals pos rate.
    reports.append(ModelReport(
        name="majority_baseline", precision=pos_rate, recall=1.0,
        f1=2 * pos_rate / (1 + pos_rate) if pos_rate else 0.0,
        pr_auc=pos_rate, accuracy=float(1 - pos_rate),
        extras={"note": "always predicts the majority class"}))

    # 1. Logistic regression, no imbalance handling.
    lr_plain = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_plain.fit(Xtr, ytr)
    models["logreg_plain"] = lr_plain
    reports.append(_metrics("logreg_plain", yte, lr_plain.predict_proba(Xte)[:, 1]))

    # 2. Logistic regression, class_weight='balanced'.
    lr_bal = LogisticRegression(max_iter=1000, class_weight="balanced",
                                random_state=RANDOM_STATE)
    lr_bal.fit(Xtr, ytr)
    models["logreg_balanced"] = lr_bal
    reports.append(_metrics("logreg_balanced", yte, lr_bal.predict_proba(Xte)[:, 1]))

    # 3. Logistic regression on oversampled training data.
    Xtr_os, ytr_os = oversample_minority(Xtr, ytr)
    lr_os = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_os.fit(Xtr_os, ytr_os)
    models["logreg_oversampled"] = lr_os
    reports.append(_metrics("logreg_oversampled", yte, lr_os.predict_proba(Xte)[:, 1]))

    # 4. LightGBM with scale_pos_weight.
    try:
        import lightgbm as lgb

        neg, pos = (ytr == 0).sum(), (ytr == 1).sum()
        gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                 scale_pos_weight=neg / max(pos, 1),
                                 random_state=RANDOM_STATE, verbose=-1)
        gbm.fit(Xtr, ytr)
        models["lightgbm_balanced"] = gbm
        reports.append(_metrics("lightgbm_balanced", yte, gbm.predict_proba(Xte)[:, 1]))
        tuned = True
    except Exception as exc:  # LightGBM optional; document the fallback
        reports.append(ModelReport(name="lightgbm_balanced", precision=float("nan"),
                                   recall=float("nan"), f1=float("nan"),
                                   pr_auc=float("nan"), accuracy=float("nan"),
                                   extras={"skipped": f"lightgbm unavailable: {exc}"}))
        tuned = False

    # 5. Threshold tuning on the best linear model (tune on train, eval on test).
    train_scores = lr_bal.predict_proba(Xtr)[:, 1]
    thr = tune_threshold(ytr, train_scores)
    models["logreg_balanced_tuned"] = lr_bal
    reports.append(_metrics("logreg_balanced_tuned", yte,
                            lr_bal.predict_proba(Xte)[:, 1], threshold=thr,
                            extras={"threshold_tuned_on": "train"}))

    return {"models": models, "reports": reports, "vectorizer": vec,
            "positive_rate": pos_rate, "n_train": len(tr_idx), "n_test": len(te_idx)}


def predict(model, texts: list[str], vectorizer=None, threshold: float = 0.5) -> list[float]:
    """Return P(market-moving-positive) per text. Vectorizer required unless
    the model was trained outside train() and handles raw text itself."""
    if vectorizer is None:
        raise ValueError("predict needs the fitted vectorizer from train()")
    X = featurize(texts, vectorizer)
    return [float(p) for p in model.predict_proba(X)[:, 1]]


def report_table(result: dict) -> pd.DataFrame:
    rows = [{
        "model": r.name, "precision": round(r.precision, 3), "recall": round(r.recall, 3),
        "f1": round(r.f1, 3), "pr_auc": round(r.pr_auc, 3),
        "accuracy": round(r.accuracy, 3), "threshold": round(r.threshold, 3),
    } for r in result["reports"]]
    return pd.DataFrame(rows)
