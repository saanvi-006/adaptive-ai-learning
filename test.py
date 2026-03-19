from app.services.document.parser import extract_text
from app.services.document.chunker import chunk_text
from app.services.embeddings.embedder import embed_text, embed_query
from app.services.embeddings.vector_store import store_embeddings, retrieve_chunks

file_path = "data/uploads/sample.pdf"

text = extract_text(file_path)
print("\n--- Extracted Text ---\n")
print(text[:500])

chunks = chunk_text(text)
print("\n--- Chunks ---\n")
print(chunks[:3])

embeddings = embed_text(chunks)
store_embeddings(chunks, embeddings)

query = "What is this document about?"
query_embedding = embed_query(query)

results = retrieve_chunks(query_embedding)

print("\n--- Retrieved Chunks ---\n")
for r in results:
    print("-", r)