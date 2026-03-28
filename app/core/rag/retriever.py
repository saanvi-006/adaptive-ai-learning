"""
RAG Retriever
app/core/rag/retriever.py

Bridges the query string to the vector store using the shared services layer.

Dependency map (DO NOT change these imports — services are fixed and deployed):
    embed_query(str)         -> app.services.embeddings.embedder
    retrieve_chunks(vec, k)  -> app.services.embeddings.vector_store
"""

from __future__ import annotations

import logging
from typing import List

from app.services.embeddings.embedder import embed_query
from app.services.embeddings.vector_store import retrieve_chunks

logger = logging.getLogger(__name__)


def retrieve(query: str, top_k: int = 3) -> List[str]:
    """
    Embed *query* with the shared embedder and return the top-k matching chunks
    from the shared vector store.

    Parameters
    ----------
    query  : Raw user question string.
    top_k  : Number of context passages to retrieve (default matches
             the services layer default of k=3).

    Returns
    -------
    List[str]  Relevant text chunks, most similar first.

    Raises
    ------
    ValueError   If *query* is empty.
    RuntimeError If the vector store has not been initialised yet
                 (store_embeddings must be called first).
    """
    if not query or not query.strip():
        raise ValueError("retrieve: query must not be empty")

    # embed_query returns a 1-D numpy array (dim,) — matches retrieve_chunks signature
    query_embedding = embed_query(query.strip())
    chunks = retrieve_chunks(query_embedding, k=top_k)

    logger.debug("retrieve(%r): returned %d chunks", query, len(chunks))
    return chunks