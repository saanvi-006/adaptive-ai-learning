from fastapi import FastAPI, UploadFile, File
import pickle
import os

from app.services.document.parser import extract_text
from app.services.document.chunker import chunk_text
from app.services.embeddings.vector_store import store_embeddings
from app.services.embeddings.embedder import embed_text
import app.core.state as state

app = FastAPI()

# ✅ In-memory storage — persists during session
_chunks_cache = []
_embeddings_cache = []

def embed_in_batches(chunks, batch_size=4):
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        emb = embed_text(batch)
        all_embeddings.extend(emb)
    return all_embeddings

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    global _chunks_cache, _embeddings_cache
    try:
        # ✅ Save to temp path
        os.makedirs("data/uploads", exist_ok=True)
        path = f"data/uploads/{file.filename}"
        with open(path, "wb") as f:
            f.write(await file.read())

        # ✅ Set state
        state.set_document(path)

        # ✅ Extract and chunk
        text = extract_text(path)
        chunks = chunk_text(text)[:5]

        if not chunks:
            return {"error": "No content"}

        # ✅ Embed
        embeddings = embed_in_batches(chunks, batch_size=4)

        # ✅ Store in memory cache
        _chunks_cache = chunks
        _embeddings_cache = embeddings

        # ✅ Load into vector store immediately
        store_embeddings(chunks, embeddings)

        return {"status": "processed", "chunks": len(chunks)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/debug-state")
def debug_state():
    import app.services.embeddings.vector_store as vs
    doc = state.get_document()
    return {
        "document_path": doc,
        "chunks_in_memory": len(_chunks_cache),
        "chunks_in_vector_store": len(vs.stored_chunks)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))