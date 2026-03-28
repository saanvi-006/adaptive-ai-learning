"""
RAG Pipeline
app/core/rag/pipeline.py

Handles CHAT queries only.
Quiz sessions must be started via quiz_engine.build_quiz_from_chunks().

Architecture:
  CHAT  → run_rag_pipeline()  → retrieve → generate_answer → adapt_response
  QUIZ  → build_quiz_from_chunks() → QuizEngine (completely separate)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

# ── Our RAG files ──────────────────────────────────────────────────────────
from app.services.document.parser import extract_text
from app.services.document.chunker import chunk_text
from app.core.rag.retriever import retrieve
from app.core.rag.generator import generate_answer

# ── Shared services (fixed, deployed — import only, do NOT modify) ─────────
from app.services.embeddings.embedder import embed_text
from app.services.embeddings.vector_store import store_embeddings
import app.services.embeddings.vector_store as _vs   # to read stored_chunks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE           = Path(__file__).resolve().parent       # app/core/rag/
_PROJECT_ROOT   = _HERE.parents[2]                      # backend/
_DEFAULT_SOURCE = str(_PROJECT_ROOT / "data" / "uploads" / "sample.pdf")

# Pipeline parameters
_CHUNK_SIZE    = 400
_CHUNK_OVERLAP = 50
_TOP_K         = 3

# Tracks which source is currently indexed
_indexed_source: str | None = None


# ---------------------------------------------------------------------------
# Public API — CHAT only
# ---------------------------------------------------------------------------

def run_rag_pipeline(
    query: str,
    source: str | None = None,
    intent: str = "factual",
    force_reindex: bool = False,
) -> dict[str, Any]:
    """
    Run the full RAG pipeline for a CHAT query.

    Intent must be "factual" or "conceptual".
    Quiz requests must use quiz_engine.build_quiz_from_chunks() instead.

    Parameters
    ----------
    query         : User question.
    source        : File path (PDF / .txt) or raw text.
    intent        : "factual" or "conceptual" — forwarded to generate_answer.
    force_reindex : Force rebuilding the index even if source is unchanged.

    Returns
    -------
    dict  {"query", "intent", "answer", "source_chunks", "num_chunks_retrieved"}
    """
    if not query or not query.strip():
        raise ValueError("run_rag_pipeline: query must not be empty")

    # Guard: quiz must never be routed through the chat pipeline
    if intent == "learning":
        raise ValueError(
            "run_rag_pipeline: intent='learning' is not valid here. "
            "Start a quiz session via quiz_engine.build_quiz_from_chunks()."
        )

    source = source or _DEFAULT_SOURCE
    _ensure_index_ready(source, force_reindex=force_reindex)

    logger.info("Retrieving top-%d chunks for: %r", _TOP_K, query)
    context_chunks = retrieve(query, top_k=_TOP_K)

    logger.info("Generating answer (%d chunks)  intent=%r", len(context_chunks), intent)
    answer = generate_answer(query, context_chunks, intent)

    return {
        "query":                query,
        "intent":               intent,
        "answer":               answer,
        "source_chunks":        context_chunks,
        "num_chunks_retrieved": len(context_chunks),
    }


def get_all_chunks(
    source: str | None = None,
    force_reindex: bool = False,
) -> list[str]:
    """
    Return ALL text chunks for the current document.

    Reads directly from vector_store.stored_chunks which is populated
    by store_embeddings() during indexing — no separate cache needed.

    Parameters
    ----------
    source        : File path (PDF / .txt). Defaults to sample.pdf.
    force_reindex : Force re-parsing even if already indexed.

    Returns
    -------
    list[str]  All text chunks from the document.
    """
    source = source or _DEFAULT_SOURCE
    _ensure_index_ready(source, force_reindex=force_reindex)

    chunks = list(_vs.stored_chunks)   # read directly from vector_store module

    if not chunks:
        raise RuntimeError(
            "get_all_chunks: vector_store.stored_chunks is empty after indexing. "
            "Check that the PDF has readable text content."
        )

    logger.info("get_all_chunks: returning %d chunks from vector_store", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Index initialisation
# ---------------------------------------------------------------------------

def _ensure_index_ready(source: str, force_reindex: bool = False) -> None:
    """
    Build the FAISS index via the services layer if not already built.

    Steps: extract_text → chunk_text → embed_text → store_embeddings
    store_embeddings() populates vector_store.stored_chunks automatically.
    """
    global _indexed_source

    if not force_reindex and _indexed_source == source:
        logger.debug("_ensure_index_ready: '%s' already indexed, skipping", source)
        return

    logger.info("Building index for source: %r", source)
    _validate_source(source)

    logger.info("  [1/3] Extracting text …")
    raw_text = extract_text(source)
    if not raw_text or not raw_text.strip():
        raise RuntimeError(f"extract_text returned empty content for: {source!r}")
    logger.info("  [1/3] %d characters extracted", len(raw_text))

    logger.info("  [2/3] Chunking (size=%d, overlap=%d) …", _CHUNK_SIZE, _CHUNK_OVERLAP)
    chunks = chunk_text(raw_text, chunk_size=_CHUNK_SIZE, overlap=_CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError("chunk_text returned zero chunks — check document content")
    logger.info("  [2/3] %d chunks created", len(chunks))

    logger.info("  [3/3] Embedding %d chunks …", len(chunks))
    embeddings = embed_text(chunks)
    logger.info("  [3/3] Embeddings shape: %s", embeddings.shape)

    # Populates both faiss index AND vector_store.stored_chunks
    store_embeddings(chunks, embeddings)

    _indexed_source = source
    logger.info("Index ready — %d chunks indexed", len(chunks))


def _validate_source(source: str) -> None:
    looks_like_path = (
        os.sep in source
        or (len(source) < 512 and "." in os.path.basename(source))
    )
    if looks_like_path and not os.path.exists(source):
        raise FileNotFoundError(
            f"Document not found: {source!r}\n"
            "Put your PDF at  data/uploads/sample.pdf  or pass raw text directly."
        )


def index_document(source: str) -> dict[str, Any]:
    """Pre-build the index for *source* at startup."""
    _ensure_index_ready(source, force_reindex=True)
    return {"source": source, "indexed": True}