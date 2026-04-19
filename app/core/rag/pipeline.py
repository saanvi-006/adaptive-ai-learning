from __future__ import annotations

import logging
from typing import Any

from app.core.rag.retriever import retrieve
from app.core.rag.generator import generate_answer
from app.core.intent.predictor import predict_intent
import app.services.embeddings.vector_store as _vs


logger = logging.getLogger(__name__)

# Retrieval parameter
_TOP_K = 3

# Tracks indexed source
_indexed_source: str | None = None


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def run_rag_pipeline(
    query: str,
    source: str | None = None,
    intent: str = "factual",
    force_reindex: bool = False,
) -> dict[str, Any]:

    if not query or not query.strip():
        raise ValueError("Query must not be empty")

    if not source:
        return {
            "query": query,
            "intent": intent,
            "answer": "No document loaded yet. Please upload a document first.",
            "source_chunks": [],
            "num_chunks_retrieved": 0
        }

    # Intent detection (safe)
    try:
        predicted_intent = predict_intent(query)
        if predicted_intent:
            intent = predicted_intent
    except Exception:
        pass

    # Ensure index is ready
    _ensure_index_ready(source, force_reindex)

    # Retrieve relevant chunks
    context_chunks = retrieve(query, top_k=_TOP_K)

    # Generate answer
    answer = generate_answer(query, context_chunks, intent)

    return {
        "query": query,
        "intent": intent,
        "answer": answer,
        "source_chunks": context_chunks,
        "num_chunks_retrieved": len(context_chunks),
    }


def get_all_chunks(
    source: str | None = None,
    force_reindex: bool = False,
) -> list[str]:

    if not source:
        return []

    _ensure_index_ready(source, force_reindex)

    chunks = list(_vs.stored_chunks)
    return chunks if chunks else []


# ---------------------------------------------------------------------------
# INTERNAL
# ---------------------------------------------------------------------------

def _ensure_index_ready(source: str, force_reindex: bool = False) -> None:
    global _indexed_source

    chunks_empty = not _vs.stored_chunks

    if not force_reindex and _indexed_source == source and not chunks_empty:
        return

    # ✅ If vector store already has chunks loaded by worker, just mark as indexed
    if _vs.stored_chunks:
        _indexed_source = source
        logger.info(f"Index already loaded with {len(_vs.stored_chunks)} chunks")
        return

    # ✅ Vector store is empty — /process hasn't been called yet
    logger.warning("Vector store is empty — /process must be called first")