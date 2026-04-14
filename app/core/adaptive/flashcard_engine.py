from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List

from google import genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-pro-latest",
]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_FLASHCARD_PROMPT_TEMPLATE = """\
Generate {count} flashcards from the content below.

RULES:
- Cover important concepts
- Questions must be clear
- Answers must be short (1–2 lines)
- No duplicate questions

OUTPUT:
Return ONLY a valid JSON array with no explanation, no markdown, no code fences.

FORMAT:
[
  {{
    "question": "...",
    "answer": "..."
  }}
]

CONTENT:
{context}
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_flashcards(
    chunks: List[str],
    *,
    target_count: int = 10,
) -> List[Dict]:

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    context = "\n\n".join(
        c.strip() for c in chunks if c and c.strip()
    )[:3000]

    if not context:
        logger.warning("No content provided to generate flashcards.")
        return []

    client = genai.Client(api_key=api_key)

    raw_text = None
    for model_name in _GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=_FLASHCARD_PROMPT_TEMPLATE.format(
                    count=target_count,
                    context=context,
                ),
            )
            if response.text:
                raw_text = response.text.strip()
                logger.info(f"Flashcard response from {model_name}")
                break
        except Exception as exc:
            logger.warning(f"Model {model_name} failed: {exc}")

    if not raw_text:
        logger.error("All Gemini models failed for flashcard generation")
        return []

    flashcards = _parse(raw_text)
    flashcards = _validate(flashcards)
    flashcards = _deduplicate(flashcards)

    return flashcards[:target_count]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(raw_text: str) -> List[Dict]:
    text = re.sub(r"```[a-zA-Z]*", "", raw_text).replace("```", "").strip()

    start = text.find("[")
    end = text.rfind("]") + 1

    if start == -1 or end == 0:
        logger.warning("No JSON array found in Gemini response.")
        logger.debug(f"Response was:\n{text[:500]}")
        return []

    json_str = text[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode failed: {e}")
        logger.debug(f"Attempted to parse:\n{json_str[:500]}")
        return []


def _validate(flashcards: List) -> List[Dict]:
    valid = []

    for card in flashcards:
        if not isinstance(card, dict):
            continue

        q = card.get("question", "").strip()
        a = card.get("answer", "").strip()

        if q and a:
            valid.append({
                "question": q,
                "answer": a[:200],
            })

    return valid


def _deduplicate(flashcards: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []

    for card in flashcards:
        key = card["question"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(card)

    return unique