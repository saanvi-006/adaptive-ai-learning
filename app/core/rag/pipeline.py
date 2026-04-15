from __future__ import annotations

import logging
import os
from typing import Any

from app.services.document.parser import extract_text
from app.services.document.chunker import chunk_text
from app.core.rag.retriever import retrieve
from app.core.rag.generator import generate_answer
from app.core.intent.predictor import predict_intent

from app.services.embeddings.embedder import embed_text
from app.services.embeddings.vector_store import store_embeddings
import app.services.embeddings.vector_store as _vs


logger = logging.getLogger(__name__)

# Pipeline parameters
_CHUNK_SIZE = 400
_CHUNK_OVERLAP = 50
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

    # ❗ REQUIRE explicit source (no default file anymore)
    if not source:
        return {
            "query": query,
            "intent": intent,
            "answer": "No document loaded yet. Please upload or select a document first.",
            "source_chunks": [],
            "num_chunks_retrieved": 0
        }

    # Safe intent override
    try:
        predicted_intent = predict_intent(query)
        if predicted_intent:
            intent = predicted_intent
    except Exception:
        pass

    # Ensure index exists
    _ensure_index_ready(source, force_reindex)

    # Retrieve + generate
    context_chunks = retrieve(query, top_k=_TOP_K)

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

    if not chunks:
        return []

    return chunks



def _ensure_index_ready(source: str, force_reindex: bool = False) -> None:
    global _indexed_source

    if not force_reindex and _indexed_source == source:
        return

    if not os.path.exists(source):
        logger.warning(f"File not found: {source}")
        return

    try:
        raw_text = extract_text(source)
        if not raw_text or not raw_text.strip():
            logger.warning("Empty document")
            return

        chunks = chunk_text(
            raw_text,
            chunk_size=_CHUNK_SIZE,
            overlap=_CHUNK_OVERLAP
        )

        if not chunks:
            logger.warning("No chunks created")
            return

        embeddings = embed_text(chunks)

        store_embeddings(chunks, embeddings)

        _indexed_source = source

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return