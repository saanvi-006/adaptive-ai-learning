from fastapi import APIRouter
import os

router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_DIR = "data/uploads"


@router.get("/")
def list_documents():
    files = os.listdir(UPLOAD_DIR)
    return {"documents": files}


@router.delete("/{filename}")
def delete_document(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(path):
        return {"error": "File not found"}

    os.remove(path)
    return {"status": "deleted", "file": filename}