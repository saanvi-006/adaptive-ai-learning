"""
Intent Classification Model — Training Script
app/core/intent/model.py

Trains a TF-IDF + Logistic Regression classifier and saves
both the vectorizer and model as a single pickle artifact.
"""

import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — resolved relative to the project root so the script works whether
# called as  `python -m app.core.intent.model`  or  `python model.py` from
# inside the package directory.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # …/app/core/intent/
_PROJECT_ROOT = _HERE.parents[2]                 # …/backend/
DATA_PATH = _PROJECT_ROOT / "data" / "uploads" / "intent_data.csv"
MODEL_DIR = _PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "intent_classifier.pkl"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_model(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Full training pipeline:
      1. Load CSV dataset
      2. Preprocess text
      3. TF-IDF vectorisation
      4. Logistic Regression training
      5. Evaluation on held-out split
      6. Persist (vectorizer, model) bundle as pickle

    Returns
    -------
    dict  Evaluation metrics produced by sklearn's classification_report
          (output_dict=True).
    """
    logger.info("Loading dataset from %s", data_path)
    df = _load_dataset(data_path)

    logger.info("Dataset loaded — %d samples across labels: %s",
                len(df), df["label"].value_counts().to_dict())

    X = df["query"].apply(_preprocess).tolist()
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Train / test split: %d / %d", len(X_train), len(X_test))

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5_000,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=1_000,
        C=1.0,
        solver="lbfgs",

        random_state=random_state,
    )
    clf.fit(X_train_vec, y_train)
    logger.info("Model training complete")

    y_pred = clf.predict(X_test_vec)
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info(
        "Evaluation:\n%s",
        classification_report(y_test, y_pred)
    )

    _save_model({"vectorizer": vectorizer, "classifier": clf}, model_path)
    logger.info("Artefact saved to %s", model_path)

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Ensure data/intent_data.csv exists in the project root."
        )
    df = pd.read_csv(path)

    required = {"query", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["query", "label"])
    df["query"] = df["query"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    valid_labels = {"factual", "conceptual", "learning"}
    invalid = set(df["label"].unique()) - valid_labels
    if invalid:
        raise ValueError(
            f"Dataset contains unexpected labels: {invalid}. "
            f"Expected: {valid_labels}"
        )

    return df


def _preprocess(text: str) -> str:
    """Lightweight text normalisation — lowercase and strip whitespace."""
    return text.lower().strip()


def _save_model(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_model()