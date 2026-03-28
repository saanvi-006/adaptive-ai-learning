from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class ExplainRequest(BaseModel):
    text: str

@router.post("/explain")
def explain_concept(request: ExplainRequest):
    try:
        import app.core.state as state
        from app.core.rag.pipeline import run_rag_pipeline
        result = run_rag_pipeline(request.text, source=state.get_document(), intent="conceptual")
        return {
            "answer": result["answer"],
            "source": result["source_chunks"]
        }
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "source": "N/A"}