import redis.asyncio as redis
from redis.exceptions import ConnectionError

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

async def safe_get(key: str):
    try:
        return await redis_client.get(key)
    except ConnectionError:
        return None  # cache miss, proceed normally

async def safe_set(key: str, value: str, ex: int = 300):
    try:
        await redis_client.set(key, value, ex=ex)
    except ConnectionError:
        pass  # skip caching silently