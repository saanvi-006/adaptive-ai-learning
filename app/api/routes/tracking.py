from fastapi import APIRouter

from app.core.adaptive.engine import get_user_performance

router = APIRouter(prefix="/tracking", tags=["Tracking"])

SESSION_KEY = "default"


@router.get("/progress")
def get_progress():
    data = get_user_performance(SESSION_KEY)
    return {"progress": data}


@router.get("/history")
def get_history():
    data = get_user_performance(SESSION_KEY)
    return {"history": data}