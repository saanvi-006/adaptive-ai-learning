from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def generate_variation(question: dict) -> dict:
    prompt = f"""You are a quiz question rephraser. Given the MCQ below, generate a variation of it.

Return ONLY a JSON object, no explanation, no markdown. Keep the same answer but rephrase the question and shuffle the options.

Format:
{{
  "question": "...",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "answer": "A",
  "explanation": "..."
}}

Original question:
{json.dumps(question)}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)

def generate_variations(questions: list) -> list:
    return [generate_variation(q) for q in questions]