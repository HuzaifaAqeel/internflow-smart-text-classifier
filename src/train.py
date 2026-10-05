"""
Train the Smart Text Classifier.

Pipeline: clean_text (FunctionTransformer) -> FeatureUnion(
    TF-IDF word 1-2 grams, TF-IDF char_wb 3-5 grams) -> classifier.

- Main model: LogisticRegression tuned with GridSearchCV (5-fold, f1_macro).
- Baseline:   MultinomialNB on the same features.
- Stratified 80/20 train/test split (seeded).
- Saves: artifacts/model.joblib (best LR pipeline),
         artifacts/vectorizer.joblib (fitted feature union),
         artifacts/metrics.json, artifacts/classification_report.txt,
         artifacts/confusion_matrix.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer  # noqa: F401  (kept for clarity)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import clean_batch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "texts.csv"
ART = ROOT / "artifacts"
RANDOM_STATE = 42
LABELS = ["Complaint", "Inquiry", "Feedback", "Other"]


def build_features() -> FeatureUnion:
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])


def build_pipeline(classifier) -> Pipeline:
    return Pipeline(
        [
            ("clean", FunctionTransformer(clean_batch, validate=False)),
            ("features", build_features()),
            ("clf", classifier),
        ]
    )


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    X = df["text"].astype(str).tolist()
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"train={len(X_train)} test={len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---- baseline: MultinomialNB -----------------------------------------
    nb = build_pipeline(MultinomialNB())
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)

    # ---- main model: tuned LogisticRegression -----------------------------
    lr = build_pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    grid = GridSearchCV(
        lr,
        param_grid={"clf__C": [0.1, 0.5, 1.0, 5.0, 10.0]},
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    lr_pred = best.predict(X_test)
    print("best params:", grid.best_params_)
    print("best CV f1_macro:", round(grid.best_score_, 4))

    metrics = {}
    for name, pred in [("naive_bayes", nb_pred), ("logistic_regression", lr_pred)]:
        metrics[name] = {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
            "f1_weighted": round(f1_score(y_test, pred, average="weighted"), 4),
        }
    metrics["best_params"] = grid.best_params_
    metrics["best_cv_f1_macro"] = round(grid.best_score_, 4)
    print(json.dumps(metrics, indent=2))

    # ---- artifacts ---------------------------------------------------------
    joblib.dump(best, ART / "model.joblib")
    joblib.dump(best.named_steps["features"], ART / "vectorizer.joblib")

    with open(ART / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(y_test, lr_pred, labels=LABELS, digits=4)
    with open(ART / "classification_report.txt", "w") as f:
        f.write("Smart Text Classifier — LogisticRegression (tuned)\n")
        f.write(f"best params: {grid.best_params_}\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, lr_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix — test set (Logistic Regression)")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(ART / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("artifacts written to", ART)


if __name__ == "__main__":
    main()
