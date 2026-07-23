"""Domain exception hierarchy, decoupled from any transport (HTTP, NSE, DB)."""


class AppError(Exception):
    """Base class for all application-level errors."""


class StockNotFoundError(AppError):
    def __init__(self, symbol: str):
        self.symbol = symbol
        super().__init__(f"Stock '{symbol}' not found")


class WatchlistNotFoundError(AppError):
    def __init__(self, watchlist_id):
        self.watchlist_id = watchlist_id
        super().__init__(f"Watchlist '{watchlist_id}' not found")


class PortfolioNotFoundError(AppError):
    def __init__(self, portfolio_id):
        self.portfolio_id = portfolio_id
        super().__init__(f"Portfolio '{portfolio_id}' not found")


class AlertNotFoundError(AppError):
    def __init__(self, alert_id):
        self.alert_id = alert_id
        super().__init__(f"Alert '{alert_id}' not found")


class NotificationNotFoundError(AppError):
    def __init__(self, notification_id):
        self.notification_id = notification_id
        super().__init__(f"Notification '{notification_id}' not found")


class EmailAlreadyRegisteredError(AppError):
    def __init__(self, email: str):
        self.email = email
        super().__init__(f"Email '{email}' is already registered")


class InvalidCredentialsError(AppError):
    def __init__(self):
        super().__init__("Invalid email or password")


class InvalidRefreshTokenError(AppError):
    def __init__(self):
        super().__init__("Invalid or expired refresh token")


class UserNotFoundError(AppError):
    """Should only happen if a valid JWT outlives its user somehow - there's
    no delete-user operation in this API, so this is a defensive guard, not
    an expected path.
    """

    def __init__(self, user_id):
        self.user_id = user_id
        super().__init__(f"User '{user_id}' not found")


class ProviderUnavailableError(AppError):
    """Raised when an upstream data provider (e.g. NSE) cannot serve a request.

    Callers in the service layer are expected to catch this and degrade
    gracefully rather than let it propagate as a 500.
    """

    def __init__(self, provider: str, reason: str):
        self.provider = provider
        self.reason = reason
        super().__init__(f"{provider} unavailable: {reason}")
