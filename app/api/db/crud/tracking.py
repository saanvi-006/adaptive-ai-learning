from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.api.db.models import Performance, QuizAttempt


async def upsert_performance(db: AsyncSession, user_id: str, snapshot: dict):
    result = await db.execute(
        select(Performance).where(Performance.user_id == user_id)
    )
    perf = result.scalar_one_or_none()

    data = {
        "correct": snapshot["correct"],
        "wrong": snapshot["wrong"],
        "total": snapshot["total"],
        "accuracy": snapshot["accuracy"],
        "last_intent": snapshot["last_intent"],
        "intent_stats": snapshot["intent_stats"],
        "weak_intents": snapshot["weak_intents"],
    }

    if not perf:
        perf = Performance(user_id=user_id, **data)
        db.add(perf)
    else:
        for k, v in data.items():
            setattr(perf, k, v)

    await db.commit()


async def save_quiz_attempt(db: AsyncSession, user_id: str, summary: dict):
    attempt = QuizAttempt(
        user_id=user_id,
        total_answered=summary["total_answered"],
        correct=summary["correct"],
        wrong=summary["wrong"],
        accuracy=summary["accuracy"],
    )
    db.add(attempt)
    await db.commit()