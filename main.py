import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.db.database import engine, Base
from app.api.routes import upload, query, explain, summarize
from app.api.routes import documents, learning, tracking, system
from app.api.routes import flashcards

api_key = os.getenv("GEMINI_API_KEY")

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(init_db())
    yield
    await engine.dispose()

app = FastAPI(
    title="BrainLoop AI Backend",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

async def init_db():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Database connected")
    except Exception as e:
        print(f"⚠️ Database error: {e}")

@app.get("/")
def home():
    return {"message": "BrainLoop backend is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}