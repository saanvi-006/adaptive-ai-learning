from fastapi import FastAPI, UploadFile, File
import pickle
import os

from app.services.document.parser import extract_text
from app.services.document.chunker import chunk_text
import app.core.state as state

app = FastAPI()

# ✅ Consistent path used everywhere
EMBEDDINGS_PATH = "embeddings.pkl"

def embed_in_batches(chunks, batch_size=4):
    from app.services.embeddings.embedder import embed_text
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
    try:
        # ✅ Save to permanent path
        os.makedirs("data/uploads", exist_ok=True)
        path = f"data/uploads/{file.filename}"

        with open(path, "wb") as f:
            f.write(await file.read())

        # ✅ Set document in state
        state.set_document(path)

        text = extract_text(path)
        chunks = chunk_text(text)[:5]

        if not chunks:
            return {"error": "No content"}

        embeddings = embed_in_batches(chunks, batch_size=4)

        # ✅ Save embeddings to consistent path
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump((chunks, embeddings), f)

        return {"status": "processed", "chunks": len(chunks)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/debug-state")
def debug_state():
    doc = state.get_document()
    import os
    embeddings_exist = os.path.exists(EMBEDDINGS_PATH)
    return {
        "document_path": doc,
        "embeddings_exist": embeddings_exist,
        "embeddings_path": EMBEDDINGS_PATH
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))