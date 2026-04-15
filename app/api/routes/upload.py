import os
import shutil
from fastapi import APIRouter, UploadFile, File
import app.core.state as state

router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # Save file only (NO heavy processing here)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Store current document reference
        state.set_document(file_path)

        return {
            "message": "PDF uploaded successfully",
            "filename": file.filename
        }

    except Exception as e:
        return {
            "message": "Upload failed",
            "error": str(e)
        }