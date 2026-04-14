import asyncio
from app.api.db.database import engine

async def test():
    async with engine.begin() as conn:
        print("✅ DB works")

asyncio.run(test())