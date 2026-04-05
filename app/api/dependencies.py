from fastapi import Depends
from sqlalchemy.orm import Session
from app.api.db.database import SessionLocal

# DB Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# (Optional placeholder for cache)
def get_cache():
    return None