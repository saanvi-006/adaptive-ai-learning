"""
Answer Generator
app/core/rag/generator.py
"""

from __future__ import annotations

import json
import logging
import os
from typing import List

logger = logging.getLogger(__name__)


_GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
    "gemini-1.0-pro",
]

# System prompt per intent
_SYSTEM_PROMPTS = {
    "factual": (
        "You are a precise educational assistant. "
        "Answer the question directly using only the provided context. "
        "Be concise and accurate."
    ),
    "conceptual": (
        "You are an educational assistant skilled at explaining concepts. "
        "Use the provided context to explain the underlying reasoning clearly. "
        "Focus on the 'why' and 'how'."
    ),
}

_DEFAULT_SYSTEM = (
    "Answer using only the provided context. Be clear and concise."
)


def generate_answer(query: str, context_chunks: List[str], intent: str = "factual") -> str:
    if not query:
        raise ValueError("generate_answer: query must not be empty")

    context = "\n\n---\n\n".join(context_chunks) if context_chunks else ""

    if intent == "learning":
        mcq = _generate_mcq(query, context)
        return _format_mcq(mcq)

    try:
        return _generate_with_gemini(query, context, intent)
    except Exception as exc:
        logger.warning("Gemini failed (%s) — using extractive fallback", exc)
        return _generate_extractive(query, context_chunks, intent)


# ---------------------------------------------------------------------------
# Gemini backend (MULTI-MODEL)
# ---------------------------------------------------------------------------

def _generate_with_gemini(query: str, context: str, intent: str) -> str:
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)

    system = _SYSTEM_PROMPTS.get(intent, _DEFAULT_SYSTEM)

    prompt = f"""
{system}

Context:
{context}

Question:
{query}

Answer:
"""

    # 🔥 Try multiple Gemini models
    for model_name in _GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)

            if response.text:
                logger.info("Generated using %s", model_name)
                return response.text.strip()

        except Exception as e:
            logger.warning("Model %s failed: %s", model_name, e)

    raise RuntimeError("All Gemini models failed")


# ---------------------------------------------------------------------------
# Extractive fallback
# ---------------------------------------------------------------------------

def _generate_extractive(query: str, chunks: List[str], intent: str) -> str:
    if not chunks:
        return f"No relevant context found for: {query}"

    query_words = set(query.lower().split())

    top_chunk = chunks[0]
    sentences = [s.strip() for s in top_chunk.split(".") if s.strip()]

    scored = sorted(
        ((len(query_words & set(s.lower().split())), i, s) for i, s in enumerate(sentences)),
        reverse=True,
    )

    best = [s for _, _, s in scored[:3]]
    base_answer = ". ".join(best) + "." if best else top_chunk[:300]

    if intent == "conceptual":
        return f"Explanation: {base_answer}"

    return base_answer


# ---------------------------------------------------------------------------
# MCQ generation (learning intent)
# ---------------------------------------------------------------------------

_MCQ_FALLBACK = {
    "question": "",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "Option A",
    "explanation": "Could not generate proper MCQ.",
}

_MCQ_PROMPT = """\
You are an educational assistant. Using ONLY the context below, generate exactly ONE multiple-choice question.

Context:
{context}

Topic: {query}

Return ONLY valid JSON:
{{
  "question": "...",
  "options": ["A", "B", "C", "D"],
  "correct_answer": "...",
  "explanation": "..."
}}"""


def _generate_mcq(query: str, context: str) -> dict:
    fallback = {**_MCQ_FALLBACK, "question": query}

    try:
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set")

        genai.configure(api_key=api_key)

        prompt = _MCQ_PROMPT.format(context=context or query, query=query)

        response = None

        # 🔥 Multi-model fallback for MCQ too
        for model_name in _GEMINI_MODELS:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)

                if response.text:
                    logger.info("MCQ generated using %s", model_name)
                    break

            except Exception as e:
                logger.warning("MCQ model %s failed: %s", model_name, e)

        if not response or not response.text:
            raise RuntimeError("All Gemini models failed")

        raw = response.text.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1].strip()

        mcq = json.loads(raw)

        return mcq

    except Exception as exc:
        logger.warning("MCQ generation failed (%s)", exc)
        return fallback


# ---------------------------------------------------------------------------
# Ensure consistent STRING output
# ---------------------------------------------------------------------------

def _format_mcq(mcq: dict) -> str:
    return f"""
Question: {mcq['question']}

A. {mcq['options'][0]}
B. {mcq['options'][1]}
C. {mcq['options'][2]}
D. {mcq['options'][3]}

Answer: {mcq['correct_answer']}

Explanation: {mcq['explanation']}
""".strip()