from sentence_transformers import SentenceTransformer

model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model   # ✅ always return

def embed_text(chunks):
    return get_model().encode(chunks)

def embed_query(query: str):
    return get_model().encode([query])[0]