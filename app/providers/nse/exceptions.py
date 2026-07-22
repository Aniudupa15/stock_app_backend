"""NSE-specific exceptions. Never allowed to leak past `nse_provider.py` —
the provider adapter translates all of these into `app.core.exceptions.ProviderUnavailableError`
(or, for a genuinely missing symbol, propagates as a normal empty/None result) before
the service layer ever sees them.
"""


class NseError(Exception):
    def __init__(self, path: str, message: str):
        self.path = path
        super().__init__(f"{path}: {message}")


class NseNotFoundError(NseError):
    """The requested resource does not exist (HTTP 404). Not a transient failure - do not retry."""

    def __init__(self, path: str):
        super().__init__(path, "not found")


class NseAuthExpiredError(NseError):
    """Session cookies were rejected (HTTP 401/403). Caller should rebootstrap and retry."""

    def __init__(self, path: str, status_code: int):
        self.status_code = status_code
        super().__init__(path, f"auth expired (HTTP {status_code})")


class NseRateLimitedError(NseError):
    """HTTP 429. Carries Retry-After (seconds) if NSE provided one."""

    def __init__(self, path: str, retry_after: float | None):
        self.retry_after = retry_after
        super().__init__(path, f"rate limited (retry_after={retry_after})")


class NseServerError(NseError):
    """HTTP 5xx or connection-level failure. Transient, safe to retry."""

    def __init__(self, path: str, status_code: int):
        self.status_code = status_code
        super().__init__(path, f"server error (HTTP {status_code})")


class NseTimeoutError(NseError):
    """Request exceeded the configured timeout. Transient, safe to retry."""

    def __init__(self, path: str):
        super().__init__(path, "timeout")
