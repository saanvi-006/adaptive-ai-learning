from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import re
import os
import app.core.state as state

from app.api.state import quiz_sessions
from app.core.rag.pipeline import get_all_chunks
from app.core.adaptive.quiz_engine import build_quiz_from_chunks

router = APIRouter(prefix="/quiz", tags=["Quiz"])

SESSION_KEY = "default"
current_questions = {}  # stores the last served question per session


# -----------------------------
# Request Models
# -----------------------------

class AnswerRequest(BaseModel):
    selected_option: str


# -----------------------------
# Helpers
# -----------------------------

def _normalize_answer(selected: str) -> str:
    """
    Normalize the selected answer to just the leading letter (A/B/C/D).

    Handles all frontend formats:
      "D. D. Informed search"  -> "D"
      "D. Informed search"     -> "D"
      "D"                      -> "D"
      "d"                      -> "D"
    """
    selected = selected.strip()
    match = re.match(r'^([A-Da-d])[\.\s]', selected)
    if match:
        return match.group(1).upper()
    # fallback: just take the first character
    return selected[0].upper()


# -----------------------------
# START QUIZ
# -----------------------------
@router.post("/start")
def start_quiz():
    source = state.get_document()
    if not source:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet. Please upload a PDF first via /upload.")

    if not os.path.exists(source):
        raise HTTPException(status_code=404, detail=f"File not found: '{source}'. Please re-upload the PDF.")

    chunks = get_all_chunks(source)
    print("Chunks received in quiz:",len(chunks))
    quiz = build_quiz_from_chunks(chunks)
    quiz_sessions[SESSION_KEY] = quiz

    first_question = quiz.get_next_question()
    current_questions[SESSION_KEY] = first_question

    return {"question": first_question}


# -----------------------------
# NEXT QUESTION
# -----------------------------
@router.get("/next")
def next_question():
    if SESSION_KEY not in quiz_sessions:
        raise HTTPException(status_code=400, detail="Quiz not started")

    quiz = quiz_sessions[SESSION_KEY]
    question = quiz.get_next_question()
    current_questions[SESSION_KEY] = question

    return {"question": question}


# -----------------------------
# SUBMIT ANSWER
# -----------------------------
@router.post("/answer")
def submit_answer(req: AnswerRequest):
    if SESSION_KEY not in quiz_sessions:
        raise HTTPException(status_code=400, detail="Quiz not started")

    question = current_questions.get(SESSION_KEY)
    if not question:
        raise HTTPException(status_code=400, detail="No active question. Call /quiz/start or /quiz/next first.")

    # Normalize whatever format the frontend sends -> single letter "A"/"B"/"C"/"D"
    normalized = _normalize_answer(req.selected_option)

    quiz = quiz_sessions[SESSION_KEY]
    result = quiz.submit_answer(
        user_id=SESSION_KEY,
        selected_answer=normalized,
        question=question,
    )
    print(f"DEBUG >>> raw selected_option: {repr(req.selected_option)}")
    print(f"DEBUG >>> correct_answer in question: {repr(question['correct_answer'])}")
    print(f"DEBUG >>> is_correct result: {result['is_correct']}")
    return result


# -----------------------------
# SUMMARY
# -----------------------------
@router.get("/summary")
def get_summary():
    if SESSION_KEY not in quiz_sessions:
        raise HTTPException(status_code=400, detail="Quiz not started")

    quiz = quiz_sessions[SESSION_KEY]
    return quiz.summary()