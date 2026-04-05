from fastapi import APIRouter
from pydantic import BaseModel

from app.core.adaptive.engine import get_user_performance

router = APIRouter(prefix="/tracking", tags=["Tracking"])


class UserRequest(BaseModel):
    user_id: str


@router.get("/progress")
def get_progress(user_id: str):
    data = get_user_performance(user_id)
    return {"progress": data}


@router.get("/history")
def get_history(user_id: str):
    data = get_user_performance(user_id)
    return {"history": data}