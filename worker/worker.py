from fastapi import FastAPI, UploadFile, File
import tempfile
import pickle

from app.services.document.parser import extract_text
from app.services.document.chunker import chunk_text

app = FastAPI()


def embed_in_batches(chunks, batch_size=4):
    # 🔥 import INSIDE function (important)
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
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            path = tmp.name

        text = extract_text(path)

        # 🔥 LIMIT chunks (CRITICAL)
        chunks = chunk_text(text)[:5]

        if not chunks:
            return {"error": "No content"}

        embeddings = embed_in_batches(chunks, batch_size=4)

        with open("embeddings.pkl", "wb") as f:
            pickle.dump((chunks, embeddings), f)

        return {"status": "processed", "chunks": len(chunks)}

    except Exception as e:
        return {"error": str(e)}
    if __name__ == "__main__":
        import uvicorn
        import os
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))