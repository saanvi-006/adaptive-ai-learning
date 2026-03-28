from fastapi import APIRouter
from app.schemas.query import QueryRequest

router = APIRouter()

@router.post("/query")
def query_system(request: QueryRequest):
    query = request.query

    try:
        from app.core.intent.predictor import predict_intent
        intent = predict_intent(query)
    except:
        if "why" in query or "explain" in query:
            intent = "conceptual"
        else:
            intent = "factual"

    try:
        import app.core.state as state
        from app.core.rag.pipeline import run_rag_pipeline
        source = state.get_document()
        result = run_rag_pipeline(query, source=source, intent=intent)
        return {
            "answer": result["answer"],
            "source": result["source_chunks"],
            "explanation": f"This is a {intent} response"
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "source": "N/A",
            "explanation": f"This is a {intent} response"
        }