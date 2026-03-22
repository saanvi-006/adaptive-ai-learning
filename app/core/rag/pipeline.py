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
from app.services.embeddings.vector_store import store_embeddings, retrieve_chunks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent      # app/core/rag/
_PROJECT_ROOT = _HERE.parents[2]             # backend/
_DEFAULT_SOURCE = str(_PROJECT_ROOT / "data" / "uploads" / "sample.pdf")

# Pipeline parameters
_CHUNK_SIZE    = 400   # words — matches teammate's chunker default
_CHUNK_OVERLAP = 50
_TOP_K         = 3     # matches services retrieve_chunks default

# Source currently held in the index (skip re-indexing on repeated queries)
_indexed_source: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_rag_pipeline(
    query: str,
    source: str | None = None,
    intent: str = "factual",
    force_reindex: bool = False,
) -> dict[str, Any]:
    """
    Run the full RAG pipeline and return a structured response.

    Parameters
    ----------
    query         : User question / request string.
    source        : File path (PDF / .txt) or raw text. Defaults to
                    ``data/uploads/sample.pdf``.
    intent        : Intent label from the classifier ("factual", "conceptual",
                    "learning"). Forwarded to generate_answer.
    force_reindex : Force rebuilding the index even if source is unchanged.

    Returns
    -------
    dict  ``{"query", "intent", "answer", "source_chunks",
             "num_chunks_retrieved"}``
    """
    if not query or not query.strip():
        raise ValueError("run_rag_pipeline: query must not be empty")

    source = source or _DEFAULT_SOURCE

    # ── STEP 1: Index must be ready BEFORE any retrieval call ─────────────
    _ensure_index_ready(source, force_reindex=force_reindex)

    # ── STEP 2: Retrieve top-k relevant chunks ─────────────────────────────
    logger.info("Retrieving top-%d chunks for: %r", _TOP_K, query)
    context_chunks = retrieve(query, top_k=_TOP_K)

    # ── STEP 3: Generate answer grounded in context ────────────────────────
    logger.info("Generating answer (%d chunks)  intent=%r", len(context_chunks), intent)
    answer = generate_answer(query, context_chunks, intent)

    return {
        "query": query,
        "intent": intent,
        "answer": answer,
        "source_chunks": context_chunks,
        "num_chunks_retrieved": len(context_chunks),
    }


# ---------------------------------------------------------------------------
# Index initialisation
# ---------------------------------------------------------------------------

def _ensure_index_ready(source: str, force_reindex: bool = False) -> None:
    """
    Build the FAISS index via the services layer if it is not already built.

    Guarantees that store_embeddings() is ALWAYS called before retrieve() so
    that retrieve_chunks() never sees a None index.

    Steps
    -----
    1. extract_text  → raw string
    2. chunk_text    → List[str]
    3. embed_text    → np.ndarray  (services embedder — fits vocabulary here)
    4. store_embeddings → populates services vector_store index
    """
    global _indexed_source

    # Fast path: same document already indexed
    if not force_reindex and _indexed_source == source:
        logger.debug("_ensure_index_ready: '%s' already indexed, skipping", source)
        return

    logger.info("Building index for source: %r", source)
    _validate_source(source)

    # 1. Extract
    logger.info("  [1/3] Extracting text …")
    raw_text = extract_text(source)
    if not raw_text or not raw_text.strip():
        raise RuntimeError(
            f"extract_text returned empty content for: {source!r}"
        )
    logger.info("  [1/3] %d characters extracted", len(raw_text))

    # 2. Chunk
    logger.info("  [2/3] Chunking (size=%d, overlap=%d) …", _CHUNK_SIZE, _CHUNK_OVERLAP)
    chunks = chunk_text(raw_text, chunk_size=_CHUNK_SIZE, overlap=_CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError("chunk_text returned zero chunks — check document content")
    logger.info("  [2/3] %d chunks created", len(chunks))

    # 3. Embed — embed_text FITS the vocabulary on this call (services layer)
    logger.info("  [3/3] Embedding %d chunks via services embedder …", len(chunks))
    embeddings = embed_text(chunks)
    logger.info("  [3/3] Embeddings shape: %s", embeddings.shape)

    # 4. Store — populates services vector_store.index (no longer None)
    store_embeddings(chunks, embeddings)
    _indexed_source = source
    logger.info("Index ready — %d chunks indexed", len(chunks))


def _validate_source(source: str) -> None:
    """Raise FileNotFoundError when source looks like a missing file path."""
    looks_like_path = (
        os.sep in source
        or (len(source) < 512 and "." in os.path.basename(source))
    )
    if looks_like_path and not os.path.exists(source):
        raise FileNotFoundError(
            f"Document not found: {source!r}\n"
            "Put your PDF at  data/uploads/sample.pdf  or pass raw text directly."
        )


# ---------------------------------------------------------------------------
# Utility: pre-index a document for repeated fast queries
# ---------------------------------------------------------------------------

def index_document(source: str) -> dict[str, Any]:
    """
    Pre-build the index for *source*.  Call this once at startup so the
    first user query is not slowed by indexing overhead.

    Returns
    -------
    dict  ``{"source": str, "num_chunks": int}``
    """
    _ensure_index_ready(source, force_reindex=True)
    return {"source": source, "indexed": True}