from __future__ import annotations

import json
import logging
import random
import re
from typing import Dict, List, Literal, Optional

from google import genai
from dotenv import load_dotenv
import os
load_dotenv()

from app.core.adaptive.engine import update_user_performance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Difficulty = Literal["easy", "medium", "hard"]
MCQ = Dict

# ---------------------------------------------------------------------------
# Difficulty ladder
# ---------------------------------------------------------------------------

_DIFFICULTY_UP: Dict[str, str] = {
    "easy":   "medium",
    "medium": "hard",
    "hard":   "hard",
}
_DIFFICULTY_DOWN: Dict[str, str] = {
    "easy":   "easy",
    "medium": "easy",
    "hard":   "medium",
}

_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-pro-latest",
]

# ---------------------------------------------------------------------------
# MCQ Generation — called ONCE per quiz session
# ---------------------------------------------------------------------------

_MCQ_PROMPT_TEMPLATE = """\
You are an expert quiz creator. Generate exactly {count} multiple-choice questions
from the content below.

STRICT RULES:
- Base every question strictly on the provided content.
- Every question must be UNIQUE. Do not repeat or rephrase any question listed under
  "Already used questions" below.
- Each MCQ must have exactly 4 options: "A. ...", "B. ...", "C. ...", "D. ..."
- correct_answer must exactly match one of the 4 option strings (e.g. "A. Paris").
- difficulty must be exactly one of: "easy", "medium", "hard"
- intent must be a short snake_case topic label inferred from the content
  (e.g. "oop", "cell_division", "conjunction") — never generic like "general".
- explanation must be exactly ONE sentence, maximum 20 words. No paragraphs. No bullets.
- Output ONLY a valid, complete JSON array. No prose. No markdown. No code fences.
- Do NOT stop early. All {count} objects must be fully closed.

Already used questions (do NOT repeat these):
{used_questions}

OUTPUT FORMAT:
[
  {{
    "question": "...",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "correct_answer": "A. ...",
    "explanation": "One sentence max 20 words.",
    "difficulty": "easy",
    "intent": "topic_slug"
  }}
]

CONTENT:
{context}"""


def generate_mcq_pool(
    context: str,
    *,
    target_count: int = 20,
) -> List[MCQ]:
    """
    Generate a pool of MCQs from *context* using Gemini.

    Generates in batches of 10 to avoid token truncation.
    Passes already-used question text to each batch so Gemini avoids repeats.
    Deduplicates by question text after each batch as a safety net.

    Parameters
    ----------
    context      : Full merged text from all PDF chunks.
    target_count : Total questions to attempt (default 20 = 2 batches of 10).

    Returns
    -------
    List[MCQ]  Validated, deduplicated list of MCQ dicts.
    """
    if not context or not context.strip():
        raise ValueError("generate_mcq_pool: context must not be empty")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    batch_size  = 10
    num_batches = max(1, target_count // batch_size)
    all_questions: List[MCQ] = []

    # Passed to each subsequent batch so Gemini avoids repeating them
    used_questions: List[str] = []

    for batch_num in range(num_batches):
        logger.info(
            "mcq_gen: batch %d/%d - requesting %d questions",
            batch_num + 1, num_batches, batch_size,
        )

        batch = _generate_batch(client, context, batch_size, used_questions)

        # Deduplicate against already collected questions by lowercased question text
        seen         = {q["question"].strip().lower() for q in all_questions}
        unique_batch = [
            q for q in batch
            if q["question"].strip().lower() not in seen
        ]

        logger.info(
            "mcq_gen: batch %d/%d - %d valid, %d unique after dedup",
            batch_num + 1, num_batches, len(batch), len(unique_batch),
        )

        all_questions.extend(unique_batch)

        # Update used list for next batch prompt
        used_questions = [q["question"] for q in all_questions]

    logger.info("mcq_gen: total unique valid questions: %d", len(all_questions))
    return all_questions


def _generate_batch(
    client,
    context: str,
    count: int,
    used_questions: List[str],
) -> List[MCQ]:
    """Call Gemini once for *count* questions. Returns validated MCQs."""
    used_block = (
        "\n".join(f"- {q}" for q in used_questions)
        if used_questions else "None yet."
    )
    prompt   = _MCQ_PROMPT_TEMPLATE.format(
        count=count,
        context=context,
        used_questions=used_block,
    )
    raw_text = None

    for model_name in _GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response.text:
                raw_text = response.text.strip()
                logger.info("mcq_gen: response from %s", model_name)
                break
        except Exception as exc:
            logger.warning("mcq_gen: model %s failed - %s", model_name, exc)

    if not raw_text:
        logger.error("mcq_gen: all Gemini models failed for this batch")
        return []

    return _parse_and_validate(raw_text)


def _parse_and_validate(raw_text: str) -> List[MCQ]:
    """
    Robustly extract MCQ objects from Gemini output.

    Strategy (in order):
      1. Strip markdown fences.
      2. Extract the outermost [...] array.
      3. Try parsing that as full JSON.
      4. If truncated/broken, salvage individual {...} objects via regex.
    """
    text = raw_text

    # Step 1: strip markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```\s*$", "", text).strip()

    # Step 2: extract outermost [...] array
    start = text.find("[")
    end   = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    # Step 3: try full JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            validated = _validate_pool(data)
            if validated:
                return validated
    except json.JSONDecodeError:
        pass

    # Step 4: regex salvage — extracts whatever complete objects exist
    # before a truncation point, instead of discarding the whole batch
    logger.warning("mcq_gen: full JSON parse failed - attempting object salvage")
    salvaged: List[MCQ] = []

    for match in re.finditer(r'\{[^{}]*"question"\s*:[^{}]*\}', text, re.DOTALL):
        try:
            obj = json.loads(match.group())
            salvaged.append(obj)
        except json.JSONDecodeError:
            continue

    if salvaged:
        logger.info("mcq_gen: salvaged %d objects via regex", len(salvaged))
        return _validate_pool(salvaged)

    logger.error("mcq_gen: could not extract any valid MCQ objects")
    return []


def _validate_pool(questions: List) -> List[MCQ]:
    """Drop any MCQ that is missing required fields or has malformed structure."""
    required = {"question", "options", "correct_answer", "explanation",
                "difficulty", "intent"}
    valid: List[MCQ] = []

    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            logger.warning("mcq_gen: item %d is not a dict - skipped", i)
            continue
        if not required.issubset(q.keys()):
            missing = required - q.keys()
            logger.warning("mcq_gen: item %d missing %s - skipped", i, missing)
            continue
        if not isinstance(q["options"], list) or len(q["options"]) != 4:
            logger.warning("mcq_gen: item %d does not have exactly 4 options - skipped", i)
            continue
        if q["difficulty"] not in ("easy", "medium", "hard"):
            logger.warning(
                "mcq_gen: item %d bad difficulty %r - defaulting to medium", i, q["difficulty"]
            )
            q["difficulty"] = "medium"
        if not str(q.get("intent", "")).strip():
            logger.warning("mcq_gen: item %d has empty intent - skipped", i)
            continue
        valid.append(q)

    return valid


# ---------------------------------------------------------------------------
# Quiz Engine
# ---------------------------------------------------------------------------

class QuizEngine:
    """
    Adaptive quiz session for a single user.

    Usage
    -----
    engine = QuizEngine(questions=generate_mcq_pool(context))
    q      = engine.get_next_question()
    result = engine.submit_answer(user_id, selected_option, q)
    """

    def __init__(self, questions: List[MCQ]) -> None:
        if not questions:
            raise ValueError("QuizEngine: questions list must not be empty")

        self.questions:          List[MCQ]  = questions
        self.current_difficulty: str        = "medium"
        self.history:            List[bool] = []
        self._asked:             set        = set()

        self._pool: Dict[str, List[int]] = {"easy": [], "medium": [], "hard": []}
        for idx, q in enumerate(questions):
            diff = q.get("difficulty", "medium")
            self._pool[diff].append(idx)

        logger.info(
            "QuizEngine: initialised  total=%d  easy=%d  medium=%d  hard=%d",
            len(questions),
            len(self._pool["easy"]),
            len(self._pool["medium"]),
            len(self._pool["hard"]),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_next_question(self) -> Optional[MCQ]:
        """
        Return the next question matching current_difficulty.
        Falls back to adjacent difficulties if the bucket is exhausted.
        Returns None when all questions have been served.
        """
        for diff in self._fallback_order():
            available = [i for i in self._pool[diff] if i not in self._asked]
            if available:
                idx = random.choice(available)
                self._asked.add(idx)
                q   = self.questions[idx]
                logger.debug(
                    "QuizEngine: serving question #%d  difficulty=%s  intent=%s",
                    idx, q["difficulty"], q["intent"],
                )
                return q

        logger.info("QuizEngine: all questions exhausted")
        return None

    def submit_answer(
        self,
        user_id:         str,
        selected_answer: str,
        question:        MCQ,
    ) -> Dict:
        """
        Process a user's answer, update engine.py, and adapt difficulty.

        Returns
        -------
        dict  {is_correct, correct_answer, explanation,
               next_difficulty, performance_snapshot}
        """
        # ✅ FIX: compare only the leading letter (A/B/C/D) to avoid
        # mismatches caused by different full-text formatting
        selected = selected_answer.strip()[0].upper()
        correct  = question["correct_answer"].strip()[0].upper()
        is_correct: bool = selected == correct

        # Delegate ALL performance tracking to engine.py
        snapshot = update_user_performance(
            user_id    = user_id,
            is_correct = is_correct,
            intent     = question["intent"],
        )

        self.history.append(is_correct)
        self._adapt_difficulty()

        logger.info(
            "QuizEngine: answer  user=%r  intent=%s  correct=%s  next_difficulty=%s",
            user_id, question["intent"], is_correct, self.current_difficulty,
        )

        return {
            "is_correct":           is_correct,
            "correct_answer":       question["correct_answer"],
            "explanation":          question["explanation"],
            "next_difficulty":      self.current_difficulty,
            "performance_snapshot": snapshot,
        }

    @property
    def questions_remaining(self) -> int:
        return len(self.questions) - len(self._asked)

    @property
    def is_complete(self) -> bool:
        return self.questions_remaining == 0

    def summary(self) -> Dict:
        total    = len(self.history)
        correct  = sum(self.history)
        accuracy = round(correct / total, 4) if total else 0.0
        return {
            "total_answered": total,
            "correct":        correct,
            "wrong":          total - correct,
            "accuracy":       accuracy,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adapt_difficulty(self) -> None:
        """
        3 correct in a row  -> increase difficulty
        2 wrong in last 3   -> decrease difficulty
        """
        last_3 = self.history[-3:]

        if len(last_3) == 3 and all(last_3):
            new_diff = _DIFFICULTY_UP[self.current_difficulty]
            if new_diff != self.current_difficulty:
                logger.debug(
                    "QuizEngine: difficulty UP  %s -> %s",
                    self.current_difficulty, new_diff,
                )
            self.current_difficulty = new_diff

        elif len(last_3) >= 2 and last_3.count(False) >= 2:
            new_diff = _DIFFICULTY_DOWN[self.current_difficulty]
            if new_diff != self.current_difficulty:
                logger.debug(
                    "QuizEngine: difficulty DOWN  %s -> %s",
                    self.current_difficulty, new_diff,
                )
            self.current_difficulty = new_diff

    def _fallback_order(self) -> List[str]:
        order = {
            "easy":   ["easy",   "medium", "hard"],
            "medium": ["medium", "easy",   "hard"],
            "hard":   ["hard",   "medium", "easy"],
        }
        return order[self.current_difficulty]


# ---------------------------------------------------------------------------
# Convenience: build a QuizEngine from PDF chunks
# ---------------------------------------------------------------------------

def build_quiz_from_chunks(
    chunks:       List[str],
    *,
    target_count: int = 20,
) -> QuizEngine:
    """
    Merge PDF chunks -> generate MCQ pool -> return a ready QuizEngine.

    Parameters
    ----------
    chunks       : All text chunks from the indexed PDF.
    target_count : Total MCQs to generate (default 20 = 2 batches of 10).

    Returns
    -------
    QuizEngine  Ready to serve questions immediately.

    Raises
    ------
    RuntimeError  If MCQ generation fails or returns no valid questions.
    """
    context = "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())

    if not context:
        raise ValueError("build_quiz_from_chunks: all chunks are empty")

    questions = generate_mcq_pool(context, target_count=target_count)

    if not questions:
        raise RuntimeError(
            "build_quiz_from_chunks: MCQ generation returned no valid questions.\n"
            "Check: GEMINI_API_KEY is set, PDF has readable text, logs above for details."
        )

    return QuizEngine(questions)