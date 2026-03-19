from app.core.rag.retriever import retrieve
from app.core.rag.generator import generate_answer

def run_rag_pipeline(query: str, intent: str):
    context = retrieve(query)
    answer = generate_answer(query, context, intent)

    return answer, context