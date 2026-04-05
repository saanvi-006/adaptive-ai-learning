# app/api/state.py

from typing import Dict
from threading import Lock

quiz_sessions: Dict[str, object] = {}
_lock = Lock()


def set_quiz(user_id: str, quiz):
    with _lock:
        quiz_sessions[user_id] = quiz


def get_quiz(user_id: str):
    return quiz_sessions.get(user_id)


def has_quiz(user_id: str) -> bool:
    return user_id in quiz_sessions


def remove_quiz(user_id: str):
    with _lock:
        quiz_sessions.pop(user_id, None)