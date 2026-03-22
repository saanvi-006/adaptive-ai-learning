"""
Hybrid Response Builder
app/core/hybrid/response_builder.py
"""

from typing import Dict, Any, List


def build_response(
    query: str,
    intent: str,
    answer: str,
    context: List[str]
) -> Dict[str, Any]:
    """
    Build a clean, structured response from RAG output.
    """

    # --- Clean answer ---
    cleaned_answer = (answer or "").strip()

    if not cleaned_answer:
        cleaned_answer = "No answer could be generated from the available content."

    # Limit length (avoid messy outputs)
    if len(cleaned_answer) > 500:
        cleaned_answer = cleaned_answer[:500].rstrip() + "..."

    # --- Context safety ---
    context = context or []
    context = context[:3]

    # --- Explanation ---
    if intent == "factual":
        explanation = "Retrieved factual answer from document context."
    elif intent == "conceptual":
        explanation = "Generated conceptual explanation using relevant context."
    elif intent == "learning":
        explanation = "Generated learning content based on retrieved material."
    else:
        explanation = "Generated response using available context."

    return {
        "query": query,
        "intent": intent,
        "answer": cleaned_answer,
        "context": context,
        "source": "rag",
        "explanation": explanation,
    }