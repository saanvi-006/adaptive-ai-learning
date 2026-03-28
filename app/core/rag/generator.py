"""
Answer Generator
app/core/rag/generator.py

Responsibilities:
  - Generate answers for CHAT queries only (factual, conceptual).
  - Quiz / MCQ generation is handled exclusively by quiz_engine.py.
  - Never receives intent="learning" — that label no longer exists in chat.
"""

from __future__ import annotations

import logging
import os
from dotenv import load_dotenv
load_dotenv()
from typing import List

logger = logging.getLogger(__name__)


_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-pro-latest",
]

# System prompt per intent
_SYSTEM_PROMPTS = {
    "factual": (
        "You are a precise educational assistant.\n"
        "Answer in 2–3 lines maximum.\n"
        "Be direct and accurate.\n"
        "Do NOT give long explanations."
    ),
    "conceptual": (
        "You are an educational assistant.\n"
        "Explain the concept clearly in a short and simple way.\n"
        "Limit to 5–6 lines maximum.\n"
        "If code is needed, give only the core snippet — no full class or boilerplate.\n"
        "Focus only on key idea."
    ),
}

_DEFAULT_SYSTEM = (
    "Answer using only the provided context. Be clear and concise."
)


def generate_answer(query: str, context_chunks: List[str], intent: str = "factual") -> str:
    """
    Generate a chat answer grounded in retrieved context.

    Parameters
    ----------
    query         : User question.
    context_chunks: Top-k chunks from the RAG retriever.
    intent        : "factual" or "conceptual" only.
                    Quiz/MCQ generation must go through quiz_engine.py.

    Returns
    -------
    str  Answer text.
    """
    if not query:
        raise ValueError("generate_answer: query must not be empty")

    # Guard: quiz intent must never reach this function
    if intent == "learning":
        raise ValueError(
            "generate_answer: intent='learning' is not valid here. "
            "MCQ generation must go through quiz_engine.py — "
            "call build_quiz_from_chunks() instead."
        )

    context = "\n\n---\n\n".join(context_chunks) if context_chunks else ""

    try:
        return _generate_with_gemini(query, context, intent)
    except Exception as exc:
        logger.warning("Gemini failed (%s) — using extractive fallback", exc)
        return _generate_extractive(query, context_chunks, intent)


# ---------------------------------------------------------------------------
# Gemini backend (multi-model with fallback chain)
# ---------------------------------------------------------------------------

def _generate_with_gemini(query: str, context: str, intent: str) -> str:
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)
    system = _SYSTEM_PROMPTS.get(intent, _DEFAULT_SYSTEM)

    prompt = f"""
{system}

Context:
{context}

Question:
{query}

Answer:
"""

    for model_name in _GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name, contents=prompt
            )
            if response.text:
                logger.info("Generated using %s", model_name)
                return response.text.strip()
        except Exception as e:
            logger.warning("Model %s failed: %s", model_name, e)

    raise RuntimeError("All Gemini models failed")


# ---------------------------------------------------------------------------
# Extractive fallback (no LLM dependency)
# ---------------------------------------------------------------------------

def _generate_extractive(query: str, chunks: List[str], intent: str) -> str:
    if not chunks:
        return f"No relevant context found for: {query}"

    query_words = set(query.lower().split())
    top_chunk   = chunks[0]
    sentences   = [s.strip() for s in top_chunk.split(".") if s.strip()]

    scored = sorted(
        ((len(query_words & set(s.lower().split())), i, s)
         for i, s in enumerate(sentences)),
        reverse=True,
    )

    best        = [s for _, _, s in scored[:3]]
    base_answer = ". ".join(best) + "." if best else top_chunk[:300]

    if intent == "conceptual":
        return f"Explanation: {base_answer}"

    return base_answer