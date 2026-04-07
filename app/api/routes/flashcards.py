from fastapi import APIRouter, HTTPException
from typing import Dict, List

import app.core.state as state

from app.core.rag.pipeline import get_all_chunks
from app.core.adaptive.flashcard_engine import generate_flashcards

router = APIRouter()


@router.post("/generate-flashcards")
async def generate_flashcards_route():
    """
    Generate flashcards from uploaded document.
    No request body required.
    """

    try:
        # 🔴 1. Get current document
        source = state.get_document()
        if not source:
            raise HTTPException(
                status_code=400,
                detail="No document uploaded. Please upload a PDF first."
            )

        # 🔴 2. Get chunks
        chunks: List[str] = get_all_chunks(source)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No content available to generate flashcards."
            )

        # 🔴 3. Generate flashcards (default = 10)
        flashcards: List[Dict] = generate_flashcards(chunks)

        if not flashcards:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate flashcards."
            )

        return {
            "flashcards": flashcards,
            "total": len(flashcards)
        }

    except HTTPException:
        raise

    except Exception as e:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Flashcard generation failed: {str(e)}"
        )