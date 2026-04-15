model = None

def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model   

def embed_text(chunks):
    return get_model().encode(chunks)

def embed_query(query: str):
    return get_model().encode([query])[0]