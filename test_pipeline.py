from app.services.document.parser import extract_text
from app.services.document.chunker import chunk_text
from app.services.embeddings.embedder import embed_text, embed_query
from app.services.embeddings.vector_store import store_embeddings, retrieve_chunks

file_path = "data/uploads/sample.pdf"

text = extract_text(file_path)


chunks = chunk_text(text)


embeddings = embed_text(chunks)
store_embeddings(chunks, embeddings)

query = "What is this document about?"
query_embedding = embed_query(query)

results = retrieve_chunks(query_embedding)

    
from app.core.routing.router import route_query
from app.core.rag.pipeline import run_rag_pipeline
from app.core.hybrid.response_builder import build_response

query = "What is method overloading?"
print(query)
intent = route_query(query)
answer, context = run_rag_pipeline(query, intent)
response = build_response(query, intent, answer, context)

print(response)