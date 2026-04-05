from app.api.db.database import AsyncSessionLocal
from app.api.db.crud.tracking import upsert_performance


async def save_performance(user_id: str, snapshot: dict):
    async with AsyncSessionLocal() as db:
        await upsert_performance(db, user_id, snapshot)