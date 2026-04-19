import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print("API KEY FOUND:", bool(api_key))

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain what is recursion in 2 lines."
)

print("\n=== GEMINI RESPONSE ===\n")
print(response.text)