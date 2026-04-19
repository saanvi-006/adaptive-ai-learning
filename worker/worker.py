from fastapi import FastAPI, UploadFile, File
import pickle
import os

from app.services.document.parser import extract_text
from app.services.document.chunker import chunk_text
import app.core.state as state

app = FastAPI()

EMBEDDINGS_PATH = "embeddings.pkl"

def embed_in_batches(chunks, batch_size=4):
    from app.services.embeddings.embedder import embed_text
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        emb = embed_text(batch)
        all_embeddings.extend(emb)
    return all_embeddings

# ✅ Auto-load embeddings into vector store on startup
@app.on_event("startup")
def load_embeddings_on_startup():
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            with open(EMBEDDINGS_PATH, "rb") as f:
                chunks, embeddings = pickle.load(f)
            from app.services.embeddings.vector_store import store_embeddings
            store_embeddings(chunks, embeddings)
            print(f"✅ Loaded {len(chunks)} chunks into vector store on startup")
        except Exception as e:
            print(f"⚠️ Could not load embeddings on startup: {e}")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    try:
        os.makedirs("data/uploads", exist_ok=True)
        path = f"data/uploads/{file.filename}"

        with open(path, "wb") as f:
            f.write(await file.read())

        state.set_document(path)

        text = extract_text(path)
        chunks = chunk_text(text)[:5]

        if not chunks:
            return {"error": "No content"}

        embeddings = embed_in_batches(chunks, batch_size=4)

        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump((chunks, embeddings), f)

        # ✅ Also load into vector store immediately after processing
        from app.services.embeddings.vector_store import store_embeddings
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
        "embeddings_exist": os.path.exists(EMBEDDINGS_PATH),
        "chunks_in_vector_store": len(vs.stored_chunks)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))