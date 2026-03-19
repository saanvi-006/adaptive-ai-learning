from app.services.embeddings.embedder import embed_query  # adjust path!
from app.services.embeddings.vector_store import retrieve_chunks  # adjust path!

def retrieve(query: str):
    query_embedding = embed_query(query)
    chunks = retrieve_chunks(query_embedding, k=3)
    return chunks