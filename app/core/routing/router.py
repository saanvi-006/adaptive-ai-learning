"""
Query Router
app/core/routing/router.py

ML-powered intent classifier → RAG pipeline → structured response.
"""

import logging
from typing import Any

from app.core.intent.predictor import predict_intent, predict_intent_with_confidence
from app.core.rag.pipeline import run_rag_pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Explanation templates per intent
# ---------------------------------------------------------------------------

_EXPLANATIONS: dict[str, str] = {
    "factual":    "Retrieved a direct answer by searching the indexed knowledge base for relevant passages.",
    "conceptual": "Constructed an explanation by retrieving context passages that cover the underlying concept.",
    "learning":   "Generated study material by retrieving relevant content and formatting it for active recall.",
}

_NO_CONTEXT_ANSWERS: dict[str, str] = {
    "factual":    "No relevant content was found in the knowledge base for this question. Try rephrasing or uploading a document that covers this topic.",
    "conceptual": "No relevant passages were found to explain this concept. Ensure a document covering this topic has been indexed.",
    "learning":   "No source material was found to generate learning content from. Please upload and index a relevant document first.",
}


# ---------------------------------------------------------------------------
# Shared handler — single implementation for all intents
# ---------------------------------------------------------------------------

def _handle(query: str, intent: str) -> dict[str, Any]:
    """Run the RAG pipeline and build a normalised response dict."""
    result = run_rag_pipeline(query=query, intent=intent)

    context = result["source_chunks"]
    n       = result["num_chunks_retrieved"]

    # Empty-context handling: surface a clear message instead of a hollow answer
    if not context or n == 0:
        return {
            "answer":      _NO_CONTEXT_ANSWERS[intent],
            "context":     [],
            "source":      "rag",
            "explanation": f"No matching passages found in the index for intent '{intent}'.",
            "context_found": False,
        }

    return {
        "answer":        result["answer"],
        "context":       context,
        "source":        "rag",
        "explanation":   f"{_EXPLANATIONS[intent]} ({n} passage{'s' if n != 1 else ''} used)",
        "context_found": True,
    }


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_ROUTE_MAP: dict[str, Any] = {
    "factual":    lambda q: _handle(q, "factual"),
    "conceptual": lambda q: _handle(q, "conceptual"),
    "learning":   lambda q: _handle(q, "learning"),
}


# ---------------------------------------------------------------------------
# Internal dispatch — shared by both public functions
# ---------------------------------------------------------------------------

def _dispatch(query: str, intent: str) -> dict[str, Any]:
    handler = _ROUTE_MAP.get(intent)
    if handler is None:
        raise RuntimeError(
            f"No handler registered for intent {intent!r}. "
            f"Registered intents: {list(_ROUTE_MAP.keys())}"
        )
    result          = handler(query)
    result["intent"] = intent
    result["query"]  = query
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route_query(query: str) -> dict[str, Any]:
    """
    Classify *query* and dispatch to the RAG pipeline.

    Returns
    -------
    dict
        ``{"query", "intent", "answer", "context", "context_found",
           "source", "explanation"}``

    Raises
    ------
    ValueError    If *query* is empty.
    RuntimeError  If the predicted intent has no registered handler.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    intent = str(predict_intent(query))
    result = _dispatch(query, intent)
    logger.info("route_query  |  intent=%r  context_found=%s  query=%r",
                intent, result.get("context_found"), query)
    return result


def route_query_verbose(query: str) -> dict[str, Any]:
    """
    Same as :func:`route_query` but adds per-class confidence scores.
    Useful for debugging and observability dashboards.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    prediction = predict_intent_with_confidence(query)
    intent     = prediction["intent"]
    result     = _dispatch(query, intent)
    result["confidence"] = prediction["confidence"]
    result["scores"]     = prediction["scores"]
    logger.info(
        "route_query_verbose  |  intent=%r  confidence=%.2f%%  context_found=%s  query=%r",
        intent, prediction["confidence"] * 100, result.get("context_found"), query,
    )
    return result