from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.api.db.models import User


async def get_or_create_user(db: AsyncSession, user_id: str):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        user = User(id=user_id)
        db.add(user)
        await db.commit()
        await db.refresh(user)

    return user