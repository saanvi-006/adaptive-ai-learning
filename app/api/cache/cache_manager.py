import json
from .redis_client import redis_client
from redis.exceptions import ConnectionError as RedisConnectionError


async def get_cache(key: str):
    try:
        data = await redis_client.get(key)
        return json.loads(data) if data else None
    except (RedisConnectionError, Exception):
        return None


async def set_cache(key: str, value, ttl=3600):
    try:
        await redis_client.set(key, json.dumps(value), ex=ttl)
    except (RedisConnectionError, Exception):
        pass


def make_key(prefix: str, **kwargs):
    return prefix + ":" + ":".join(f"{k}={v}" for k, v in kwargs.items())