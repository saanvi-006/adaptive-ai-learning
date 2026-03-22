"""
Intent Classification — Inference Module
app/core/intent/predictor.py

Loads the trained artefact once (module-level singleton) and exposes
`predict_intent(query)` for use by the router and any other caller.
"""

import logging
import pickle
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution — works from any working directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # …/app/core/intent/
_PROJECT_ROOT = _HERE.parents[2]                 # …/backend/
_MODEL_PATH = _PROJECT_ROOT / "models" / "intent_classifier.pkl"

IntentLabel = Literal["factual", "conceptual", "learning"]


# ---------------------------------------------------------------------------
# Lazy singleton loader
# ---------------------------------------------------------------------------

class _ModelRegistry:
    """Thread-safe lazy loader that holds the vectorizer + classifier pair."""

    def __init__(self) -> None:
        self._vectorizer = None
        self._classifier = None
        self._loaded = False

    def _load(self, model_path: Path = _MODEL_PATH) -> None:
        if self._loaded:
            return

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artefact not found at '{model_path}'. "
                "Run `python -m app.core.intent.model` from the project root "
                "to train and save the model first."
            )

        logger.info("Loading intent classifier from %s", model_path)
        with open(model_path, "rb") as fh:
            bundle = pickle.load(fh)

        required_keys = {"vectorizer", "classifier"}
        missing = required_keys - set(bundle.keys())
        if missing:
            raise KeyError(
                f"Model bundle is missing keys: {missing}. "
                "Re-train the model using app/core/intent/model.py."
            )

        self._vectorizer = bundle["vectorizer"]
        self._classifier = bundle["classifier"]
        self._loaded = True
        logger.info("Intent classifier loaded successfully")

    # Public accessors — trigger load on first access
    @property
    def vectorizer(self):
        self._load()
        return self._vectorizer

    @property
    def classifier(self):
        self._load()
        return self._classifier


_registry = _ModelRegistry()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_intent(query: str) -> IntentLabel:
    """
    Classify *query* into one of three intent categories.

    Parameters
    ----------
    query : str
        Raw user query string.

    Returns
    -------
    str
        One of ``"factual"``, ``"conceptual"``, or ``"learning"``.

    Raises
    ------
    FileNotFoundError
        If the model artefact has not been created yet.
    ValueError
        If *query* is empty or not a string.
    """
    if not isinstance(query, str):
        raise ValueError(f"query must be a str, got {type(query).__name__!r}")
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    normalised = query.lower()
    features = _registry.vectorizer.transform([normalised])
    label: str = _registry.classifier.predict(features)[0]
    logger.debug("predict_intent(%r) -> %r", query, label)
    return label  # type: ignore[return-value]


def predict_intent_with_confidence(query: str) -> dict:
    """
    Extended variant that also returns per-class probability scores.

    Returns
    -------
    dict
        ``{"intent": str, "confidence": float, "scores": dict[str, float]}``
    """
    if not isinstance(query, str):
        raise ValueError(f"query must be a str, got {type(query).__name__!r}")
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    normalised = query.lower()
    features = _registry.vectorizer.transform([normalised])
    label: str = _registry.classifier.predict(features)[0]
    proba = _registry.classifier.predict_proba(features)[0]
    classes = _registry.classifier.classes_

    scores = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    confidence = scores[label]

    logger.debug(
        "predict_intent_with_confidence(%r) -> %r (%.2f%%)",
        query, label, confidence * 100
    )
    return {"intent": label, "confidence": confidence, "scores": scores}


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        "Why do plants need sunlight?",
        "What is photosynthesis?",
        "Generate quiz on photosynthesis",
    ]
    for s in samples:
        print(f"{s!r:55s}  →  {predict_intent(s)}")