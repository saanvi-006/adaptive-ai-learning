import os
import shutil
import httpx
import asyncio
from fastapi import APIRouter, UploadFile, File
import app.core.state as state

router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ✅ FIXED: correct endpoint
WORKER_URL = "https://adaptive-ai-learning-3.onrender.com/process"


# 🔥 background worker call
async def call_worker(file_path: str, filename: str, content_type: str):
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            with open(file_path, "rb") as f:
                response = await client.post(
                    WORKER_URL,
                    files={"file": (filename, f, content_type)}
                )
                print("Worker response:", response.status_code)
    except Exception as e:
        print(f"Worker call failed: {e}")


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # Save locally
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        state.set_document(file_path)

        # 🔥 NON-BLOCKING
        asyncio.create_task(
            call_worker(file_path, file.filename, file.content_type)
        )

        return {
            "message": "Uploaded successfully. Processing in background."
        }

    except Exception as e:
        return {
            "message": "Upload failed",
            "error": str(e)
        }