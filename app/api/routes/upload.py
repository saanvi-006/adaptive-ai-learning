from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()

    # TEMP fallback (until Member 2 connects)
    try:
        from app.services.document.parser import extract_text
        from app.services.document.chunker import chunk_text
        from app.services.embeddings.embedder import generate_embeddings
        from app.services.embeddings.vector_store import store_embeddings

        text = extract_text(content)
        chunks = chunk_text(text)
        embeddings = generate_embeddings(chunks)
        store_embeddings(chunks, embeddings)

        return {"message": "PDF processed successfully"}

    except Exception as e:
        # fallback so your API doesn't break
        return {
            "message": "PDF received (processing pending)",
            "error": str(e)
        }