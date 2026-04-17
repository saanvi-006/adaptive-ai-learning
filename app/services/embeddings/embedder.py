import os
import requests

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.environ.get("HF_TOKEN")  # add this in Render env vars

def embed_text(chunks):
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": chunks}
    )
    return response.json()

def embed_query(query: str):
    return embed_text([query])[0]