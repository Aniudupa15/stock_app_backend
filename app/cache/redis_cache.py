import logging
import pickle

import redis.asyncio as redis
from redis.exceptions import RedisError

from app.domain.ports import CachePort

logger = logging.getLogger(__name__)


class RedisCache(CachePort):
    """Redis-backed CachePort. Values are pickled - cache contents are
    server-internal (never user-controllable input, e.g. never deserialized
    from a request body), so pickle's usual deserialization risk doesn't
    apply here; it's the simplest way to round-trip arbitrary cached Python
    objects (Pydantic model instances, lists of them) without writing a
    custom encoder for every cached shape.

    Fails open: every method catches `RedisError` and degrades to a no-op
    (`get` returns None, `set`/`delete` are swallowed) rather than letting a
    Redis outage turn a cached endpoint into a 500 - matching the same
    graceful-degradation principle used for NSE provider failures.
    """

    def __init__(self, redis_url: str):
        self._client = redis.from_url(redis_url)

    async def get(self, key: str) -> object | None:
        try:
            raw = await self._client.get(key)
        except RedisError as exc:
            logger.warning("Redis GET failed for key=%s, treating as cache miss: %s", key, exc)
            return None
        if raw is None:
            return None
        try:
            return pickle.loads(raw)
        except (pickle.PickleError, EOFError) as exc:
            logger.warning("Redis value for key=%s failed to unpickle, treating as cache miss: %s", key, exc)
            return None

    async def set(self, key: str, value: object, ttl_seconds: int) -> None:
        try:
            await self._client.set(key, pickle.dumps(value), ex=ttl_seconds)
        except RedisError as exc:
            logger.warning("Redis SET failed for key=%s, continuing without caching it: %s", key, exc)

    async def delete(self, key: str) -> None:
        try:
            await self._client.delete(key)
        except RedisError as exc:
            logger.warning("Redis DELETE failed for key=%s: %s", key, exc)

    async def aclose(self) -> None:
        await self._client.aclose()
