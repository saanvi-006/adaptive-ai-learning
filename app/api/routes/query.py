from fastapi import APIRouter, BackgroundTasks

from app.schemas.query import QueryRequest

from app.api.cache.cache_manager import get_cache, set_cache, make_key
from app.api.tasks.performance_task import save_performance

from app.core.adaptive.engine import adapt_response, get_user_performance
import traceback

router = APIRouter()

SESSION_KEY = "default"

@router.post("/query")
async def query_system(request: QueryRequest, background_tasks: BackgroundTasks):
    query = request.query

    # -------------------------
    # CACHE CHECK
    # -------------------------
    cache_key = make_key("query", q=query)

    cached = await get_cache(cache_key)
    if cached:
        return cached

    # -------------------------
    # Intent detection
    # -------------------------
    try:
        from app.core.intent.predictor import predict_intent
        intent = predict_intent(query)
    except:
        if "why" in query or "explain" in query:
            intent = "conceptual"
        else:
            intent = "factual"

    # -------------------------
    # RAG pipeline
    # -------------------------
    try:
        import app.core.state as state
        from app.core.rag.pipeline import run_rag_pipeline

        source = state.get_document()

        result = run_rag_pipeline(
            query,
            source=source,
            intent=intent
        )

        # -------------------------
        # ADAPTIVE RESPONSE
        # -------------------------
        adapted_answer = adapt_response(
            user_id=SESSION_KEY,
            intent=intent,
            answer=result["answer"]
        )

        response = {
            "answer": adapted_answer,
            "source": result["source_chunks"],
            "intent": intent
        }

        # -------------------------
        # SAVE PERFORMANCE (ASYNC SAFE)
        # -------------------------
        snapshot = get_user_performance(SESSION_KEY)
        background_tasks.add_task(save_performance, SESSION_KEY, snapshot)

        # -------------------------
        # CACHE STORE
        # -------------------------
        await set_cache(cache_key, response)

        return response

    except Exception as e:
        traceback.print_exc()
        return {
            "answer": f"Error: {str(e)}",
            "source": "N/A",
            "intent": intent
        }