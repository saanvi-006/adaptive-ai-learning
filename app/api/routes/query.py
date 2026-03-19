from fastapi import APIRouter
from app.schemas.query import QueryRequest

router = APIRouter()

@router.post("/query")
def query_system(request: QueryRequest):
    query = request.query

    # TEMP fallback intent
    try:
        from app.core.intent.predictor import predict_intent
        intent = predict_intent(query)
    except:
        if "why" in query or "explain" in query:
            intent = "conceptual"
        else:
            intent = "factual"

    # TEMP fallback RAG
    try:
        from app.core.rag.pipeline import rag_pipeline
        rag_result = rag_pipeline(query)
    except:
        rag_result = {
            "answer": "Sample answer (RAG not connected yet)",
            "source": "N/A"
        }

    # TEMP fallback response builder
    try:
        from app.core.hybrid.response_builder import build_response
        response = build_response(query, rag_result, intent)
    except:
        response = {
            "answer": rag_result.get("answer"),
            "source": rag_result.get("source"),
            "explanation": f"This is a {intent} response"
        }

    return response