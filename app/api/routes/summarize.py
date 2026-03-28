from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class SummarizeRequest(BaseModel):
    document_id: str = None

@router.post("/summarize")
def summarize_document(request: SummarizeRequest):
    try:
        import app.core.state as state
        from app.core.rag.pipeline import run_rag_pipeline
        result = run_rag_pipeline(
            "Summarize the main topics and key points of this document.",
            source=state.get_document(),
            intent="conceptual"
        )
        return {
            "answer": result["answer"],
            "source": result["source_chunks"]
        }
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "source": "N/A"}