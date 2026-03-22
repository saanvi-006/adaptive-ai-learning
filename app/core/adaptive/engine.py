from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCURACY_LOW:   float = 0.4   # below -> struggling tier
ACCURACY_HIGH:  float = 0.7   # above -> proficient tier
WEAK_THRESHOLD: float = 0.5   # per-intent accuracy below this -> weak intent

_PREFIX_SIMPLIFY = "Let's simplify this: "
_SUFFIX_ADVANCED = (
    "\n\nAdvanced Insight: This topic has deeper layers worth exploring. "
    "Consider looking into edge cases, underlying theory, or real-world "
    "applications to strengthen your understanding."
)
_SUFFIX_PRACTICE = "\n\n📌 You should practice more on this topic."

_NO_HISTORY_TIER = "developing"   # brand-new users start at mid-tier


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _IntentStat:
    correct: int = 0
    total:   int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


@dataclass
class _UserRecord:
    correct:      int = 0
    wrong:        int = 0
    last_intent:  str = ""
    intent_stats: Dict[str, _IntentStat] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.correct + self.wrong

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def weak_intents(self) -> List[str]:
        return [
            intent for intent, stat in self.intent_stats.items()
            if stat.total > 0 and stat.accuracy < WEAK_THRESHOLD
        ]

    def to_dict(self) -> dict:
        """Serialise to a plain dict (safe for JSON / API responses)."""
        return {
            "correct":      self.correct,
            "wrong":        self.wrong,
            "total":        self.total,
            "accuracy":     round(self.accuracy, 4),
            "last_intent":  self.last_intent,
            "weak_intents": self.weak_intents,
            "intent_stats": {
                intent: {
                    "correct":  s.correct,
                    "total":    s.total,
                    "accuracy": round(s.accuracy, 4),
                }
                for intent, s in self.intent_stats.items()
            },
        }


# ---------------------------------------------------------------------------
# In-memory store  (thread-safe)
# ---------------------------------------------------------------------------

_lock:  threading.Lock          = threading.Lock()
_store: Dict[str, _UserRecord]  = {}


def _get_or_create(user_id: str) -> _UserRecord:
    if user_id not in _store:
        _store[user_id] = _UserRecord()
        logger.debug("engine: created record  user=%r", user_id)
    return _store[user_id]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update_user_performance(user_id: str, is_correct: bool, intent: str) -> dict:
    """
    Record one answer event for *user_id*.

    Parameters
    ----------
    user_id    : Unique identifier for the learner.
    is_correct : Whether the learner answered correctly.
    intent     : Intent category of the question.

    Returns
    -------
    dict  Updated performance snapshot.
    """
    if not user_id:
        raise ValueError("update_user_performance: user_id must not be empty")
    if not intent:
        raise ValueError("update_user_performance: intent must not be empty")

    with _lock:
        record = _get_or_create(user_id)

        if is_correct:
            record.correct += 1
        else:
            record.wrong += 1

        if intent not in record.intent_stats:
            record.intent_stats[intent] = _IntentStat()
        stat = record.intent_stats[intent]
        if is_correct:
            stat.correct += 1
        stat.total += 1

        record.last_intent = intent
        snapshot = record.to_dict()

    logger.info(
        "engine: update  user=%r  intent=%r  correct=%s  "
        "global_accuracy=%.2f  weak=%s",
        user_id, intent, is_correct,
        snapshot["accuracy"], snapshot["weak_intents"],
    )
    return snapshot


def get_user_performance(user_id: str) -> dict:
    """
    Return the current performance snapshot for *user_id*.

    Returns an all-zero record if the user has no history yet.
    """
    with _lock:
        return _get_or_create(user_id).to_dict()


def reset_user(user_id: str) -> None:
    """Wipe all performance data for *user_id*."""
    with _lock:
        _store.pop(user_id, None)
    logger.info("engine: reset  user=%r", user_id)


def adapt_response(
    user_id: str,
    intent:  str,
    answer:  Union[str, dict],
) -> Union[str, dict]:
    """
    Adjust *answer* based on the learner's current performance.

    Dict answers (MCQ / structured content) are returned unchanged — they
    already carry structured content that should not be modified.

    Parameters
    ----------
    user_id : Learner identifier.
    intent  : Intent label for this interaction.
    answer  : Raw answer from the generator (str or dict).

    Returns
    -------
    str | dict  Adapted answer.
    """
    if not user_id or not intent:
        return answer

    # Structured responses are not adapted
    if isinstance(answer, dict):
        logger.debug(
            "engine: adapt skipped (dict)  user=%r  intent=%r", user_id, intent
        )
        return answer

    with _lock:
        record       = _get_or_create(user_id)
        accuracy     = record.accuracy
        no_history   = record.total == 0
        weak_intents = record.weak_intents

    tier    = _accuracy_tier(accuracy, no_history)
    adapted = _apply_tier(answer, tier)
    adapted = _apply_weak_flag(adapted, intent, weak_intents)

    logger.debug(
        "engine: adapt  user=%r  intent=%r  tier=%s  "
        "accuracy=%.2f  weak_flag=%s",
        user_id, intent, tier, accuracy, intent in weak_intents,
    )
    return adapted


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _accuracy_tier(accuracy: float, no_history: bool) -> str:
    if no_history:
        return _NO_HISTORY_TIER
    if accuracy < ACCURACY_LOW:
        return "struggling"
    if accuracy > ACCURACY_HIGH:
        return "proficient"
    return "developing"


def _apply_tier(answer: str, tier: str) -> str:
    if tier == "struggling":
        return _PREFIX_SIMPLIFY + answer
    if tier == "proficient":
        return answer + _SUFFIX_ADVANCED
    return answer


def _apply_weak_flag(answer: str, intent: str, weak_intents: List[str]) -> str:
    if intent in weak_intents:
        return answer + _SUFFIX_PRACTICE
    return answer