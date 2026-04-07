from pydantic import BaseModel
from typing import List, Dict, Optional

class FlashcardResponse(BaseModel):
    flashcards: List[Dict]
    total: int