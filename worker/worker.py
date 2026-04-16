from fastapi import FastAPI, UploadFile, File
import tempfile
import pickle

from app.services.document.parser import extract_text
from app.services.document.chunker import chunk_text
from app.services.embeddings.embedder import embed_text

app = FastAPI()


def embed_in_batches(chunks, batch_size=8):
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        emb = embed_text(batch)
        all_embeddings.extend(emb)

    return all_embeddings


@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            path = tmp.name

        # Extract + chunk
        text = extract_text(path)
        chunks = chunk_text(text)

        if not chunks:
            return {"error": "No content"}

        # 🔥 KEY FIX: batch embedding
        embeddings = embed_in_batches(chunks, batch_size=8)

        # Save
        with open("embeddings.pkl", "wb") as f:
            pickle.dump((chunks, embeddings), f)

        return {"status": "processed", "chunks": len(chunks)}

    except Exception as e:
        return {"error": str(e)}