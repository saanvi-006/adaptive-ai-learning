from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.state import quiz_sessions

from app.core.rag.pipeline import get_all_chunks
from app.core.adaptive.quiz_engine import build_quiz_from_chunks

router = APIRouter(prefix="/quiz", tags=["Quiz"])


# -----------------------------
# Request Models
# -----------------------------

class StartQuizRequest(BaseModel):
    user_id: str
    source: str  # PDF path


class AnswerRequest(BaseModel):
    user_id: str
    selected_option: str


# -----------------------------
# START QUIZ
# -----------------------------
@router.post("/start")
def start_quiz(req: StartQuizRequest):
    user_id = req.user_id

    # Step 1: get chunks
    chunks = get_all_chunks(req.source)

    # Step 2: build quiz (ONLY ONCE)
    quiz = build_quiz_from_chunks(chunks)

    # Step 3: store session
    quiz_sessions[user_id] = quiz

    # Step 4: first question
    first_question = quiz.get_next_question()

    return {"question": first_question}


# -----------------------------
# NEXT QUESTION
# -----------------------------
@router.get("/next")
def next_question(user_id: str):
    if user_id not in quiz_sessions:
        raise HTTPException(status_code=400, detail="Quiz not started")

    quiz = quiz_sessions[user_id]
    question = quiz.get_next_question()

    return {"question": question}


# -----------------------------
# SUBMIT ANSWER
# -----------------------------
@router.post("/answer")
def submit_answer(req: AnswerRequest):
    user_id = req.user_id

    if user_id not in quiz_sessions:
        raise HTTPException(status_code=400, detail="Quiz not started")

    quiz = quiz_sessions[user_id]

    result = quiz.submit_answer(
        user_id=user_id,
        selected_answer=req.selected_option
    )

    return result


# -----------------------------
# SUMMARY
# -----------------------------
@router.get("/summary")
def get_summary(user_id: str):
    if user_id not in quiz_sessions:
        raise HTTPException(status_code=400, detail="Quiz not started")

    quiz = quiz_sessions[user_id]
    summary = quiz.get_summary()

    return summary