import faiss
import numpy as np

index = None
stored_chunks = []


def store_embeddings(chunks, embeddings):
    global index, stored_chunks

    if not embeddings or len(embeddings) == 0:
        return

    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    stored_chunks = chunks


def retrieve_chunks(query_embedding, k=3):
    global index, stored_chunks

    if index is None:
        return []

    query_embedding = np.array([query_embedding]).astype("float32")

    D, I = index.search(query_embedding, k)

    return [stored_chunks[i] for i in I[0] if i < len(stored_chunks)]