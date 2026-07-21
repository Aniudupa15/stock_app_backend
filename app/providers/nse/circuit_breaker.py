"""Minimal asyncio-native circuit breaker.

`pybreaker`'s async support (`call_async`) is built on Tornado's legacy
`gen.coroutine`, not native asyncio, and errors out without Tornado
installed - a poor fit for an asyncio/FastAPI stack. The classic breaker
state machine (closed -> open -> half-open) is small enough to implement
directly rather than pull in Tornado as a transitive dependency for it.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


class CircuitOpenError(Exception):
    def __init__(self):
        super().__init__("circuit breaker is open")


class AsyncCircuitBreaker:
    def __init__(self, fail_max: int, reset_timeout_seconds: float, exclude: tuple[type[Exception], ...] = ()):
        self._fail_max = fail_max
        self._reset_timeout = reset_timeout_seconds
        self._exclude = exclude
        self._failures = 0
        self._state = "closed"  # closed | open | half_open
        self._opened_at = 0.0
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
        async with self._lock:
            if self._state == "open":
                if time.monotonic() - self._opened_at >= self._reset_timeout:
                    self._state = "half_open"
                else:
                    raise CircuitOpenError

        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            if not isinstance(exc, self._exclude):
                async with self._lock:
                    self._failures += 1
                    if self._state == "half_open" or self._failures >= self._fail_max:
                        self._state = "open"
                        self._opened_at = time.monotonic()
            raise
        else:
            async with self._lock:
                self._failures = 0
                self._state = "closed"
            return result
