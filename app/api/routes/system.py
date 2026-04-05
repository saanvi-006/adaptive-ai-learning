from fastapi import APIRouter
from pydantic import BaseModel

from app.core.intent.predictor import predict_intent

router = APIRouter(prefix="/system", tags=["System"])


class IntentRequest(BaseModel):
    query: str


@router.post("/intent")
def get_intent(req: IntentRequest):
    intent = predict_intent(req.query)
    return {"intent": intent}