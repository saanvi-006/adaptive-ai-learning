import os
import shutil
from fastapi import APIRouter, UploadFile, File
import app.core.state as state
import app.core.state as state
router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        state.set_document(file_path)

        from app.services.document.parser import extract_text
        from app.services.document.chunker import chunk_text
        from app.services.embeddings.embedder import embed_text
        from app.services.embeddings.vector_store import store_embeddings

        text = extract_text(file_path)
        chunks = chunk_text(text)
        embeddings = embed_text(chunks)
        store_embeddings(chunks, embeddings)

        return {"message": "PDF processed successfully", "filename": file.filename}

    except Exception as e:
        return {
            "message": "PDF received (processing pending)",
            "error": str(e)
        }