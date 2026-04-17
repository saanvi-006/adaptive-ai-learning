import os
import requests
import time

API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
HF_TOKEN = os.environ.get("HF_TOKEN")

def embed_text(chunks, retries=3):
    for i in range(retries):
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": chunks},
            timeout=30
        )

        if response.status_code == 200:
            return response.json()

        if response.status_code == 503:
            time.sleep(10)
            continue

        raise Exception(f"HF API error {response.status_code}: {response.text}")

    raise Exception("Failed after retries")

def embed_query(query: str):
    return embed_text([query])[0]