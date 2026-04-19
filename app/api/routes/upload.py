import os
import shutil
import httpx
import asyncio
from fastapi import APIRouter, UploadFile, File
import app.core.state as state

router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

WORKER_URL = "https://adaptive-ai-learning-3.onrender.com/process"


async def call_worker_and_load(file_path: str, filename: str, content_type: str):
    try:
        async with httpx.AsyncClient(timeout=120) as client:

            # STEP 1: Send file to worker
            with open(file_path, "rb") as f:
                response = await client.post(
                    WORKER_URL,
                    files={"file": (filename, f, content_type)}
                )

            print("Worker response:", response.status_code)
            data = response.json()
            print("Worker data:", data)

            # STEP 2: Fetch embeddings (IMPORTANT: still inside client block)
            if data.get("status") == "processed":
                embed_response = await client.get(
                    "https://adaptive-ai-learning-3.onrender.com/get-embeddings"
                )

                print("Embedding fetch status:", embed_response.status_code)

                if embed_response.status_code == 200:
                    payload = embed_response.json()

                    chunks = payload.get("chunks", [])
                    embeddings = payload.get("embeddings", [])

                    print("Received chunks:", len(chunks))

                    if chunks and embeddings:
                        from app.services.embeddings.vector_store import store_embeddings
                        store_embeddings(chunks, embeddings)
                        print(f"✅ Loaded {len(chunks)} chunks into main app vector store")
                    else:
                        print("⚠️ Empty embeddings received")

    except Exception as e:
        print(f"Worker call failed: {e}")


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        state.set_document(file_path)

        await call_worker_and_load(file_path, file.filename, file.content_type)

        return {"message": "Uploaded successfully. Processing in background."}

    except Exception as e:
        return {"message": "Upload failed", "error": str(e)}