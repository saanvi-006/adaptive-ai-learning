from fastapi import FastAPI
from app.api.routes import upload, query

# Create FastAPI app
app = FastAPI(
    title="Adaptive AI Backend",
    version="1.0.0"
)

# Include routers
app.include_router(upload.router)
app.include_router(query.router)

# Root endpoint (for testing)
@app.get("/")
def home():
    return {"message": "Backend is running 🚀"}