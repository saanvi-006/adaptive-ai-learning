import json
from .redis_client import redis_client


async def get_cache(key: str):
    data = await redis_client.get(key)
    return json.loads(data) if data else None


async def set_cache(key: str, value, ttl=3600):
    await redis_client.set(key, json.dumps(value), ex=ttl)


def make_key(prefix: str, **kwargs):
    return prefix + ":" + ":".join(f"{k}={v}" for k, v in kwargs.items())