import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.db.database import engine, Base 
from app.api.routes import upload, query, explain, summarize
from app.api.routes import documents, learning, tracking, system
from app.api.routes import flashcards

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

app = FastAPI(
    title="BrainLoop AI Backend",
    version="1.0.0"
)

# ✅ FIXED: Removed space from URL
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "https://id-preview--35605c8b-1541-4a1a-9885-f30a2373d7e5.lovable.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, tags=["Documents"])
app.include_router(flashcards.router, tags=["Flashcards"])
app.include_router(query.router, tags=["AI Interaction"])
app.include_router(explain.router, tags=["AI Interaction"])
app.include_router(summarize.router, tags=["AI Interaction"])

app.include_router(documents.router, tags=["Documents"])
app.include_router(learning.router, tags=["Quiz"])
app.include_router(tracking.router, tags=["Tracking"])
app.include_router(system.router, tags=["System"])

# ✅ FIXED: Added try/except so app starts even if DB fails
@app.on_event("startup")  
async def startup():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Database connected")
    except Exception as e:
        print(f"⚠️ Database error (app will still start): {e}")

@app.get("/")
def home():
    return {"message": "BrainLoop backend is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}