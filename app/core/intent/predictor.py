"""
Intent Classification — Inference Module
app/core/intent/predictor.py

Classifies CHAT queries into: "factual" or "conceptual".

"learning" is NOT a chat intent. Quiz sessions are triggered explicitly
by the user pressing "Start Quiz" — not by intent classification.
"""

import logging
import pickle
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_HERE         = Path(__file__).resolve().parent    # …/app/core/intent/
_PROJECT_ROOT = _HERE.parents[2]                   # …/backend/
_MODEL_PATH   = _PROJECT_ROOT / "models" / "intent_classifier.pkl"

# Only two valid chat intents
IntentLabel = Literal["factual", "conceptual"]


# ---------------------------------------------------------------------------
# Lazy singleton loader
# ---------------------------------------------------------------------------

class _ModelRegistry:
    """Thread-safe lazy loader that holds the vectorizer + classifier pair."""

    def __init__(self) -> None:
        self._vectorizer = None
        self._classifier = None
        self._loaded     = False

    def _load(self, model_path: Path = _MODEL_PATH) -> None:
        if self._loaded:
            return

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artefact not found at '{model_path}'. "
                "Run `python -m app.core.intent.model` to train the model first."
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
        self._loaded     = True
        logger.info("Intent classifier loaded successfully")

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
    Classify a CHAT query into "factual" or "conceptual".

    Quiz sessions are not classified — they are triggered explicitly
    by the user and handled by quiz_engine.py.

    Parameters
    ----------
    query : Raw user query string.

    Returns
    -------
    str  "factual" or "conceptual".
    """
    if not isinstance(query, str):
        raise ValueError(f"query must be a str, got {type(query).__name__!r}")
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    normalised = query.lower()
    features   = _registry.vectorizer.transform([normalised])
    label: str = _registry.classifier.predict(features)[0]

    # Safety net: if old model still returns "learning", remap to conceptual
    if label == "learning":
        logger.warning(
            "predict_intent: got deprecated label 'learning' — "
            "remapping to 'conceptual'. Re-train the model to fix this."
        )
        label = "conceptual"

    logger.debug("predict_intent(%r) -> %r", query, label)
    return label  # type: ignore[return-value]


def predict_intent_with_confidence(query: str) -> dict:
    """
    Extended variant that also returns per-class probability scores.

    Returns
    -------
    dict  {"intent": str, "confidence": float, "scores": dict[str, float]}
    """
    if not isinstance(query, str):
        raise ValueError(f"query must be a str, got {type(query).__name__!r}")
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    normalised = query.lower()
    features   = _registry.vectorizer.transform([normalised])
    label: str = _registry.classifier.predict(features)[0]
    proba      = _registry.classifier.predict_proba(features)[0]
    classes    = _registry.classifier.classes_

    # Safety net for old model artefacts
    if label == "learning":
        logger.warning(
            "predict_intent_with_confidence: deprecated label 'learning' — "
            "remapping to 'conceptual'. Re-train the model to fix this."
        )
        label = "conceptual"

    scores     = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    confidence = scores.get(label, 0.0)

    logger.debug(
        "predict_intent_with_confidence(%r) -> %r (%.2f%%)",
        query, label, confidence * 100,
    )
    return {"intent": label, "confidence": confidence, "scores": scores}


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        "Why do plants need sunlight?",
        "What is photosynthesis?",
    ]
    for s in samples:
        print(f"{s!r:55s}  →  {predict_intent(s)}")