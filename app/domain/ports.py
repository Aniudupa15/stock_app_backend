"""Abstract interfaces (ports) that the service layer depends on.

Concrete adapters (NSE HTTP client, SQLAlchemy repository, in-memory cache)
implement these. Nothing in `services/` imports a concrete adapter directly;
wiring happens once in `app/api/deps.py`. This is what makes the data
provider and cache swappable without touching business logic.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import date, datetime
from decimal import Decimal

from app.domain.entities import (
    Alert,
    AlertStatus,
    AlertType,
    BhavcopyRecord,
    CorporateAction,
    FinancialResultFiling,
    FinancialResultRecord,
    IndexQuote,
    IntradaySignalSnapshot,
    IpoFiling,
    LongTermSignalSnapshot,
    MarketMover,
    MarketStatus,
    NewsArticle,
    NewsCategory,
    Notification,
    OhlcvBar,
    Portfolio,
    PortfolioTransaction,
    Quote,
    RefreshToken,
    ScreenerFilters,
    SearchHistoryEntry,
    Stock,
    StockIndicatorSnapshot,
    StockMasterRecord,
    User,
    Watchlist,
    WatchlistItem,
)


class StockDataProviderPort(ABC):
    """Live/external market data."""

    @abstractmethod
    async def get_quote(self, symbol: str) -> Quote:
        """Fetch a live quote for a single symbol. Raises ProviderUnavailableError on failure."""

    @abstractmethod
    async def fetch_equity_universe(self) -> list[StockMasterRecord]:
        """Fetch the full listed-equity master list. Raises ProviderUnavailableError on failure."""

    @abstractmethod
    async def fetch_daily_bars(self, trade_date: date) -> list[BhavcopyRecord]:
        """Fetch one day's full-market Bhavcopy, filtered to equity rows.
        Raises ProviderUnavailableError on failure (including "no file for this
        date" being surfaced distinctly - callers should treat holidays/weekends
        as a normal, expected empty result, not a hard failure).
        """

    @abstractmethod
    async def fetch_corporate_actions(self, from_date: date, to_date: date) -> list[CorporateAction]:
        """Fetch corporate actions (dividends/splits/bonuses/etc.) in a date window.
        Raises ProviderUnavailableError on failure - this endpoint is cookie-gated
        and less reliable than the static archive-backed methods above.
        """

    @abstractmethod
    async def fetch_financial_results_index(
        self, symbol: str, from_date: date, to_date: date
    ) -> list[FinancialResultFiling]:
        """List of quarterly financial-results filings for a symbol in a date
        window (metadata only - no parsed figures yet). Filters to the recent
        "Ind-AS New" taxonomy and filings with a usable XBRL attachment; older
        filings are silently excluded, not errored. Raises ProviderUnavailableError
        on failure - cookie-gated, same reliability class as corporate actions.
        """

    @abstractmethod
    async def fetch_financial_result_detail(self, filing: FinancialResultFiling) -> FinancialResultRecord | None:
        """Fetch and parse one filing's XBRL document into quarter figures.
        Returns None if the filing can't be confidently parsed (unexpected
        structure, no matching duration context) - not an error, since this is
        expected to happen for a minority of filings. Raises
        ProviderUnavailableError only for genuine fetch failures.
        """

    @abstractmethod
    async def fetch_market_status(self) -> list[MarketStatus]:
        """Best-effort - cookie-gated, same reliability class as get_quote.
        Raises ProviderUnavailableError on failure; callers should degrade
        gracefully (e.g. omit the field) rather than fail the whole request.
        """

    @abstractmethod
    async def fetch_indices(self) -> list[IndexQuote]:
        """Best-effort - cookie-gated, same reliability class as get_quote.
        Raises ProviderUnavailableError on failure; callers should degrade
        gracefully rather than fail the whole request.
        """

    @abstractmethod
    async def fetch_ipo_filings(self) -> list[IpoFiling]:
        """Merges two NSE endpoints (active/upcoming issues + already-listed
        past issues) into one list. Best-effort per sub-endpoint - if one of
        the two fails, still returns whatever the other one gave; only
        raises ProviderUnavailableError if both fail.
        """


class StockRepositoryPort(ABC):
    """Persistence for the `stocks` table."""

    @abstractmethod
    async def get_by_symbol(self, symbol: str) -> Stock | None: ...

    @abstractmethod
    async def search_by_symbol_or_name(self, query: str, limit: int) -> list[Stock]: ...

    @abstractmethod
    async def upsert_universe(self, records: list[StockMasterRecord]) -> int:
        """Idempotently upsert master records; returns the number of rows affected."""

    @abstractmethod
    async def deactivate_missing(self, active_symbols: set[str]) -> int:
        """Soft-delist any stock not present in `active_symbols`; returns count deactivated."""

    @abstractmethod
    async def list_active_symbols(self) -> list[str]:
        """Every active stock's symbol - used to tag news articles with
        related_symbols without a live NSE call.
        """


class HistoricalPriceRepositoryPort(ABC):
    """Persistence for the `historical_prices` table."""

    @abstractmethod
    async def bulk_upsert_bars(self, records: list[BhavcopyRecord]) -> int:
        """Idempotently upsert daily bars (unmatched symbols are skipped, not errored);
        returns the number of rows upserted.
        """

    @abstractmethod
    async def get_bars(self, symbol: str, from_date: date, to_date: date) -> list[OhlcvBar]:
        """Bars for a symbol in [from_date, to_date], ascending by trade_date."""


class CorporateActionRepositoryPort(ABC):
    """Persistence for the `corporate_actions` table."""

    @abstractmethod
    async def bulk_upsert(self, records: list[CorporateAction]) -> int: ...

    @abstractmethod
    async def get_for_symbol(self, symbol: str) -> list[CorporateAction]: ...

    @abstractmethod
    async def list_dividend_actions(self, ex_date_from: date, ex_date_to: date) -> list[CorporateAction]:
        """Every corporate action across all active stocks whose `purpose`
        mentions "dividend" and whose `ex_date` falls in
        [ex_date_from, ex_date_to] - powers the cross-stock dividend list
        without a per-symbol loop over the whole universe.
        """


class FinancialResultRepositoryPort(ABC):
    """Persistence for the `financial_results` table."""

    @abstractmethod
    async def bulk_upsert(self, records: list[FinancialResultRecord]) -> int: ...

    @abstractmethod
    async def get_recent_quarters(self, symbol: str, consolidated: bool, limit: int) -> list[FinancialResultRecord]:
        """Most recent `limit` quarters for a symbol, descending by period_end,
        filtered to the given consolidated/standalone flag (mixing the two
        would corrupt growth-rate math).
        """


class MarketMoverRepositoryPort(ABC):
    """Read-only leaderboards derived from `historical_prices` - no writes,
    no new table (it's a query, not a fact).
    """

    @abstractmethod
    async def get_top_movers(self, direction: str, lookback_sessions: int, limit: int) -> list[MarketMover]:
        """Top gainers (`direction="gainers"`) or losers (`"losers"`) ranked by
        percent change from `lookback_sessions` trading sessions ago to the
        latest close. Stocks without enough price history for the requested
        lookback are excluded (not enough data to rank them for that period).
        """

    @abstractmethod
    async def get_most_active(self, limit: int) -> list[MarketMover]:
        """Ranked by latest session's volume, descending. `change`/`change_percent`
        reflect the latest 1-day move and are `None` for stocks with fewer
        than 2 sessions of history.
        """

    @abstractmethod
    async def get_52_week_extremes(self, direction: str, limit: int) -> list[MarketMover]:
        """Stocks whose latest close is a new 252-session high (`direction="high"`)
        or low (`"low"`). `change`/`change_percent` reflect the latest 1-day move.
        """

    @abstractmethod
    async def get_latest_prices(self, symbols: list[str]) -> list[MarketMover]:
        """Latest close + 1-day change for exactly the given symbols (order not
        guaranteed) - used to compose watchlist/portfolio views without a live
        NSE call per holding. Symbols with no price history are simply absent
        from the result, not errored.
        """


class WatchlistRepositoryPort(ABC):
    """Persistence for `watchlists` and `watchlist_items`."""

    @abstractmethod
    async def create(self, user_id: uuid.UUID, name: str) -> Watchlist: ...

    @abstractmethod
    async def list_for_user(self, user_id: uuid.UUID) -> list[Watchlist]: ...

    @abstractmethod
    async def get(self, watchlist_id: uuid.UUID, user_id: uuid.UUID) -> Watchlist | None:
        """Returns None if the watchlist doesn't exist OR belongs to a different
        user - callers can't distinguish "not found" from "not yours", which is
        the correct behavior (don't leak existence of other users' data).
        """

    @abstractmethod
    async def delete(self, watchlist_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """Returns True if a row was deleted (owned by user_id and existed)."""

    @abstractmethod
    async def add_item(self, watchlist_id: uuid.UUID, symbol: str) -> bool:
        """Idempotent - adding an already-present symbol is a no-op, still
        returns True. Returns False if `symbol` matches no known stock
        (caller should raise StockNotFoundError; this port has no dependency
        on StockRepositoryPort to check that itself).
        """

    @abstractmethod
    async def remove_item(self, watchlist_id: uuid.UUID, symbol: str) -> bool:
        """Returns True if the item existed and was removed."""

    @abstractmethod
    async def get_items(self, watchlist_id: uuid.UUID) -> list[WatchlistItem]: ...


class PortfolioRepositoryPort(ABC):
    """Persistence for `portfolios` and `portfolio_transactions`. Holdings
    (avg cost, qty, P&L) are never stored - they're derived on read from the
    transaction log, so there's nothing to keep in sync.
    """

    @abstractmethod
    async def create(self, user_id: uuid.UUID, name: str) -> Portfolio: ...

    @abstractmethod
    async def list_for_user(self, user_id: uuid.UUID) -> list[Portfolio]: ...

    @abstractmethod
    async def get(self, portfolio_id: uuid.UUID, user_id: uuid.UUID) -> Portfolio | None:
        """Returns None if the portfolio doesn't exist OR belongs to a
        different user (same not-found-vs-not-yours non-distinction as
        WatchlistRepositoryPort.get).
        """

    @abstractmethod
    async def add_transaction(self, portfolio_id: uuid.UUID, transaction: PortfolioTransaction) -> bool:
        """Returns False if `transaction.symbol` matches no known stock
        (caller should raise StockNotFoundError)."""

    @abstractmethod
    async def get_transactions(self, portfolio_id: uuid.UUID) -> list[PortfolioTransaction]:
        """All transactions for a portfolio, ascending by transaction_date."""


class NewsProviderPort(ABC):
    """External news source (RSS feeds, not NSE - a second provider vertical)."""

    @abstractmethod
    async def fetch_latest(self, known_symbols: set[str]) -> list[NewsArticle]:
        """Fetch and parse every configured feed. `known_symbols` is used to
        tag each article's `related_symbols` (substring/word match against
        headline+summary) - passed in rather than looked up internally, since
        providers don't depend on repositories. Raises ProviderUnavailableError
        only if every configured feed failed; a partial fetch (some feeds
        down) still returns whatever succeeded.
        """


class NewsRepositoryPort(ABC):
    """Persistence for `news_articles`."""

    @abstractmethod
    async def bulk_upsert(self, articles: list[NewsArticle]) -> int:
        """Idempotent on `url` - re-syncing the same article is a no-op."""

    @abstractmethod
    async def list_latest(
        self, category: NewsCategory | None, symbol: str | None, limit: int, offset: int
    ) -> list[NewsArticle]:
        """Most recent articles first, optionally filtered by category and/or
        a symbol appearing in `related_symbols`.
        """


class AlertRepositoryPort(ABC):
    """Persistence for `alerts`. Every alert is stock-scoped in this phase -
    the schema's `stock_id` column is nullable to leave room for future
    portfolio-wide alerts, but nothing here creates or evaluates one yet.
    """

    @abstractmethod
    async def create(self, user_id: uuid.UUID, symbol: str, alert_type: AlertType, condition: dict) -> Alert | None:
        """Returns None if `symbol` matches no known stock (caller should
        raise StockNotFoundError)."""

    @abstractmethod
    async def list_for_user(self, user_id: uuid.UUID, status: AlertStatus | None) -> list[Alert]: ...

    @abstractmethod
    async def get(self, alert_id: uuid.UUID, user_id: uuid.UUID) -> Alert | None: ...

    @abstractmethod
    async def delete(self, alert_id: uuid.UUID, user_id: uuid.UUID) -> bool: ...

    @abstractmethod
    async def list_active(self) -> list[Alert]:
        """Every ACTIVE alert across all users - consumed by the evaluation engine."""

    @abstractmethod
    async def mark_triggered(self, alert_id: uuid.UUID, triggered_at: datetime) -> None: ...


class NotificationRepositoryPort(ABC):
    """Persistence for `notifications`. In-app only in this phase - push/email
    need external accounts (Firebase/APNs, SMTP) only the user can set up.
    """

    @abstractmethod
    async def create(
        self, user_id: uuid.UUID, alert_id: uuid.UUID | None, title: str, message: str
    ) -> Notification: ...

    @abstractmethod
    async def list_for_user(
        self, user_id: uuid.UUID, unread_only: bool, limit: int, offset: int
    ) -> list[Notification]: ...

    @abstractmethod
    async def mark_read(self, notification_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """Returns True if the notification existed (and belonged to user_id)."""


class UserRepositoryPort(ABC):
    """Persistence for `users`. Nothing outside the auth service/repository
    layer needs this port - every other feature works with a bare user_id.
    """

    @abstractmethod
    async def get_by_email(self, email: str) -> User | None: ...

    @abstractmethod
    async def get_by_id(self, user_id: uuid.UUID) -> User | None: ...

    @abstractmethod
    async def create(self, email: str, password_hash: str, display_name: str) -> User: ...

    @abstractmethod
    async def update(self, user_id: uuid.UUID, display_name: str | None, email: str | None) -> User:
        """Partial update - only the non-None fields are changed. Callers
        must have already confirmed `user_id` exists (e.g. via a JWT) and,
        if changing email, that the new email isn't already taken by a
        different user - this method assumes both are already validated.
        """


class RefreshTokenRepositoryPort(ABC):
    """Persistence for `refresh_tokens`. Tokens are looked up and revoked by
    hash - the raw token value is never stored, matching password hashing.
    """

    @abstractmethod
    async def create(self, user_id: uuid.UUID, token_hash: str, expires_at: datetime) -> RefreshToken: ...

    @abstractmethod
    async def get_by_hash(self, token_hash: str) -> RefreshToken | None: ...

    @abstractmethod
    async def revoke(self, token_id: uuid.UUID) -> None: ...


class SearchHistoryRepositoryPort(ABC):
    """Persistence for `search_history`. Logging is best-effort from the
    caller's side (a failure here must never break a search) - this port
    itself just does the write/read/delete, the "never fail the search"
    guarantee lives in the service that calls it.
    """

    @abstractmethod
    async def log(self, user_id: uuid.UUID, query: str) -> None: ...

    @abstractmethod
    async def list_for_user(self, user_id: uuid.UUID, limit: int, offset: int) -> list[SearchHistoryEntry]:
        """Most recent first."""

    @abstractmethod
    async def clear_for_user(self, user_id: uuid.UUID) -> int:
        """Returns the number of rows deleted."""


class ScreenerRepositoryPort(ABC):
    """Persistence for `stock_indicator_snapshots` - written by the daily
    snapshot sync job, read by the screener. Unlike every other repository
    in this app, there's no live/on-demand computation path: filtering
    happens entirely against yesterday's materialized snapshot.
    """

    @abstractmethod
    async def bulk_upsert(self, snapshots: list[StockIndicatorSnapshot]) -> int:
        """One row per stock (unique on stock_id) - each sync run overwrites
        the previous day's values rather than accumulating history.
        """

    @abstractmethod
    async def screen(self, filters: ScreenerFilters, limit: int) -> list[StockIndicatorSnapshot]: ...


class IntradaySignalSnapshotRepositoryPort(ABC):
    """Persistence for `intraday_signal_snapshots` - written by the daily
    signal snapshot sync job (which reuses `IntradaySignalService.get_signal()`
    across every active stock, not a separate scoring path), read by the
    Analysis screen's intraday tab. Same materialized-snapshot rationale as
    `ScreenerRepositoryPort`.
    """

    @abstractmethod
    async def bulk_upsert(self, snapshots: list[IntradaySignalSnapshot]) -> int:
        """One row per stock (unique on stock_id) - each sync run overwrites
        the previous day's values rather than accumulating history.
        """

    @abstractmethod
    async def list_top(self, min_confidence: Decimal, limit: int) -> list[IntradaySignalSnapshot]:
        """BUY/SELL signals only (HOLD is excluded - "top picks" isn't
        meaningful for a fence-sitting call), ordered by confidence descending.
        """


class LongTermSignalSnapshotRepositoryPort(ABC):
    """Persistence for `long_term_signal_snapshots` - same rationale as
    `IntradaySignalSnapshotRepositoryPort`.
    """

    @abstractmethod
    async def bulk_upsert(self, snapshots: list[LongTermSignalSnapshot]) -> int: ...

    @abstractmethod
    async def list_top(self, min_confidence: int, tenure: str | None, limit: int) -> list[LongTermSignalSnapshot]:
        """BUY signals only, optionally filtered to one `investment_tenure`
        bucket, ordered by confidence descending.
        """


class IpoRepositoryPort(ABC):
    """Persistence for `ipo_filings`, unique on `symbol` - a sync run
    overwrites a company's row in place (e.g. Active -> Listed) rather than
    accumulating a new row per status change.
    """

    @abstractmethod
    async def bulk_upsert(self, filings: list[IpoFiling]) -> int: ...

    @abstractmethod
    async def list_all(self, status: str | None, limit: int, offset: int) -> list[IpoFiling]:
        """Most recently synced first."""


class CachePort(ABC):
    @abstractmethod
    async def get(self, key: str) -> object | None: ...

    @abstractmethod
    async def set(self, key: str, value: object, ttl_seconds: int) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...
