from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def generate_flashcards(chunks: list, num_cards: int = 5) -> list:
    context = "\n".join(chunks[:5])
    prompt = f"""You are a flashcard generator. Based on the content below, generate {num_cards} flashcards.

Return ONLY a JSON array, no explanation, no markdown. Format:
[
  {{
    "question": "...",
    "answer": "..."
  }}
]

Content:
{context}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)