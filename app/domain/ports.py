"""Abstract interfaces (ports) that the service layer depends on.

Concrete adapters (NSE HTTP client, SQLAlchemy repository, in-memory cache)
implement these. Nothing in `services/` imports a concrete adapter directly;
wiring happens once in `app/api/deps.py`. This is what makes the data
provider and cache swappable without touching business logic.
"""

from abc import ABC, abstractmethod
from datetime import date

from app.domain.entities import (
    BhavcopyRecord,
    CorporateAction,
    FinancialResultFiling,
    FinancialResultRecord,
    OhlcvBar,
    Quote,
    Stock,
    StockMasterRecord,
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
    async def fetch_financial_results_index(self, symbol: str, from_date: date, to_date: date) -> list[FinancialResultFiling]:
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


class CachePort(ABC):
    @abstractmethod
    async def get(self, key: str) -> object | None: ...

    @abstractmethod
    async def set(self, key: str, value: object, ttl_seconds: int) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...
