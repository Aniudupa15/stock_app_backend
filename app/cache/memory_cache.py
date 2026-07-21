import asyncio
import time

from app.domain.ports import CachePort


class _Entry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: object, expires_at: float):
        self.value = value
        self.expires_at = expires_at


class InMemoryCache(CachePort):
    """Async-safe in-memory TTL cache. Single-process only - fine for Phase 1's
    low-volume search/quote caching. Swap for a Redis-backed CachePort implementation
    once multi-process deployment or shared cache state is actually needed.
    """

    def __init__(self):
        self._store: dict[str, _Entry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> object | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at < time.monotonic():
                del self._store[key]
                return None
            return entry.value

    async def set(self, key: str, value: object, ttl_seconds: int) -> None:
        async with self._lock:
            self._store[key] = _Entry(value, time.monotonic() + ttl_seconds)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)
