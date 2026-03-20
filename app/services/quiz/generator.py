from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def generate_quiz(chunks: list, num_questions: int = 5) -> list:
    context = "\n".join(chunks[:5])
    prompt = f"""You are a quiz generator. Based on the content below, generate {num_questions} MCQ questions.

Return ONLY a JSON array, no explanation, no markdown. Format:
[
  {{
    "question": "...",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "answer": "A",
    "explanation": "..."
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