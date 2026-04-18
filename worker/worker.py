from fastapi import FastAPI, UploadFile, File
import tempfile
import pickle
import os

from app.services.document.parser import extract_text
from app.services.document.chunker import chunk_text
import app.core.state as state

app = FastAPI()

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
        # ✅ Save to permanent path (not tempfile) so state can find it later
        os.makedirs("data/uploads", exist_ok=True)
        path = f"data/uploads/{file.filename}"
        
        with open(path, "wb") as f:
            f.write(await file.read())

        # ✅ Set document in state so /generate-flashcards and /quiz/start work
        state.set_document(path)

        text = extract_text(path)
        chunks = chunk_text(text)[:5]

        if not chunks:
            return {"error": "No content"}

        embeddings = embed_in_batches(chunks, batch_size=4)

        with open("embeddings.pkl", "wb") as f:
            pickle.dump((chunks, embeddings), f)

        return {"status": "processed", "chunks": len(chunks)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
@app.get("/debug-state")
def debug_state():
    import app.core.state as state
    doc = state.get_document()
    return {"document_path": doc}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))