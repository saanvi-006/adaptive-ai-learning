from app.core.routing.router import route_query
from app.core.rag.pipeline import run_rag_pipeline
from app.core.hybrid.response_builder import build_response

query = "Why do plants need sunlight?"

intent = route_query(query)
answer, context = run_rag_pipeline(query, intent)
response = build_response(query, intent, answer, context)

print(response)