"""from fastapi import FastAPI, UploadFile, File
import tempfile
import pickle

from app.services.document.parser import extract_text
from app.services.document.chunker import chunk_text
from app.services.embeddings.embedder import embed_text

app = FastAPI()


@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            path = tmp.name

        # Process
        text = extract_text(path)
        chunks = chunk_text(text)[:50]  # limit

        embeddings = embed_text(chunks)

        # Save embeddings (IMPORTANT)
        with open("embeddings.pkl", "wb") as f:
            pickle.dump((chunks, embeddings), f)

        return {"status": "processed"}

    except Exception as e:
        return {"error": str(e)}
        """
    

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}