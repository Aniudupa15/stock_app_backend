"""Domain exception hierarchy, decoupled from any transport (HTTP, NSE, DB)."""


class AppError(Exception):
    """Base class for all application-level errors."""


class StockNotFoundError(AppError):
    def __init__(self, symbol: str):
        self.symbol = symbol
        super().__init__(f"Stock '{symbol}' not found")


class ProviderUnavailableError(AppError):
    """Raised when an upstream data provider (e.g. NSE) cannot serve a request.

    Callers in the service layer are expected to catch this and degrade
    gracefully rather than let it propagate as a 500.
    """

    def __init__(self, provider: str, reason: str):
        self.provider = provider
        self.reason = reason
        super().__init__(f"{provider} unavailable: {reason}")
