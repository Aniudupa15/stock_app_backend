import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.domain.entities import (
    BhavcopyRecord,
    CorporateAction,
    FinancialResultFiling,
    FinancialResultRecord,
    InstrumentType,
    OhlcvBar,
    Quote,
    Stock,
    StockMasterRecord,
)
from app.domain.ports import (
    CachePort,
    CorporateActionRepositoryPort,
    FinancialResultRepositoryPort,
    HistoricalPriceRepositoryPort,
    StockDataProviderPort,
    StockRepositoryPort,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeCache(CachePort):
    def __init__(self):
        self.store: dict[str, object] = {}

    async def get(self, key: str) -> object | None:
        return self.store.get(key)

    async def set(self, key: str, value: object, ttl_seconds: int) -> None:
        self.store[key] = value

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)


class FakeStockRepository(StockRepositoryPort):
    def __init__(self, stocks: list[Stock] | None = None):
        self.stocks = {s.symbol: s for s in (stocks or [])}
        self.upserted: list[StockMasterRecord] = []
        self.deactivated_calls: list[set[str]] = []

    async def get_by_symbol(self, symbol: str) -> Stock | None:
        return self.stocks.get(symbol.strip().upper())

    async def search_by_symbol_or_name(self, query: str, limit: int) -> list[Stock]:
        q = query.strip().lower()
        matches = [
            s for s in self.stocks.values() if q in s.symbol.lower() or q in s.name.lower()
        ]
        return matches[:limit]

    async def upsert_universe(self, records: list[StockMasterRecord]) -> int:
        self.upserted.extend(records)
        for r in records:
            self.stocks[r.symbol] = Stock(
                symbol=r.symbol,
                isin=r.isin,
                name=r.name,
                series=r.series,
                sector=None,
                industry=None,
                instrument_type=InstrumentType.EQUITY,
                listing_date=r.listing_date,
                face_value=r.face_value,
                is_active=True,
            )
        return len(records)

    async def deactivate_missing(self, active_symbols: set[str]) -> int:
        self.deactivated_calls.append(active_symbols)
        count = 0
        for symbol, stock in list(self.stocks.items()):
            if symbol not in active_symbols and stock.is_active:
                self.stocks[symbol] = replace(stock, is_active=False)
                count += 1
        return count


class FakeStockDataProvider(StockDataProviderPort):
    def __init__(
        self,
        quotes: dict[str, Quote] | None = None,
        universe: list[StockMasterRecord] | None = None,
        daily_bars: list[BhavcopyRecord] | None = None,
        corporate_actions: list[CorporateAction] | None = None,
        financial_filings: list[FinancialResultFiling] | None = None,
        financial_details: dict[str, FinancialResultRecord] | None = None,
    ):
        self.quotes = quotes or {}
        self.universe = universe or []
        self.daily_bars = daily_bars or []
        self.corporate_actions = corporate_actions or []
        self.financial_filings = financial_filings or []
        self.financial_details = financial_details or {}  # keyed by xbrl_url
        self.fail_with: Exception | None = None

    async def get_quote(self, symbol: str) -> Quote:
        if self.fail_with:
            raise self.fail_with
        if symbol not in self.quotes:
            raise KeyError(symbol)
        return self.quotes[symbol]

    async def fetch_equity_universe(self) -> list[StockMasterRecord]:
        if self.fail_with:
            raise self.fail_with
        return self.universe

    async def fetch_daily_bars(self, trade_date: date) -> list[BhavcopyRecord]:
        if self.fail_with:
            raise self.fail_with
        return self.daily_bars

    async def fetch_corporate_actions(self, from_date: date, to_date: date) -> list[CorporateAction]:
        if self.fail_with:
            raise self.fail_with
        return self.corporate_actions

    async def fetch_financial_results_index(
        self, symbol: str, from_date: date, to_date: date
    ) -> list[FinancialResultFiling]:
        if self.fail_with:
            raise self.fail_with
        return self.financial_filings

    async def fetch_financial_result_detail(self, filing: FinancialResultFiling) -> FinancialResultRecord | None:
        if self.fail_with:
            raise self.fail_with
        return self.financial_details.get(filing.xbrl_url)


class FakeHistoricalPriceRepository(HistoricalPriceRepositoryPort):
    def __init__(self, bars: dict[str, list[OhlcvBar]] | None = None):
        self.bars_by_symbol: dict[str, list[OhlcvBar]] = bars or {}
        self.upserted: list[BhavcopyRecord] = []

    async def bulk_upsert_bars(self, records: list[BhavcopyRecord]) -> int:
        self.upserted.extend(records)
        for r in records:
            bar = OhlcvBar(trade_date=r.trade_date, open=r.open, high=r.high, low=r.low, close=r.close, volume=r.volume)
            self.bars_by_symbol.setdefault(r.symbol, []).append(bar)
        return len(records)

    async def get_bars(self, symbol: str, from_date: date, to_date: date) -> list[OhlcvBar]:
        bars = self.bars_by_symbol.get(symbol.strip().upper(), [])
        return sorted((b for b in bars if from_date <= b.trade_date <= to_date), key=lambda b: b.trade_date)


class FakeCorporateActionRepository(CorporateActionRepositoryPort):
    def __init__(self, actions: dict[str, list[CorporateAction]] | None = None):
        self.actions_by_symbol: dict[str, list[CorporateAction]] = actions or {}

    async def bulk_upsert(self, records: list[CorporateAction]) -> int:
        for r in records:
            self.actions_by_symbol.setdefault(r.symbol, []).append(r)
        return len(records)

    async def get_for_symbol(self, symbol: str) -> list[CorporateAction]:
        return self.actions_by_symbol.get(symbol.strip().upper(), [])


class FakeFinancialResultRepository(FinancialResultRepositoryPort):
    def __init__(self, quarters: dict[str, list[FinancialResultRecord]] | None = None):
        self.quarters_by_symbol: dict[str, list[FinancialResultRecord]] = quarters or {}

    async def bulk_upsert(self, records: list[FinancialResultRecord]) -> int:
        for r in records:
            self.quarters_by_symbol.setdefault(r.symbol, []).append(r)
        return len(records)

    async def get_recent_quarters(self, symbol: str, consolidated: bool, limit: int) -> list[FinancialResultRecord]:
        quarters = [
            q for q in self.quarters_by_symbol.get(symbol.strip().upper(), []) if q.consolidated == consolidated
        ]
        quarters.sort(key=lambda q: q.period_end, reverse=True)
        return quarters[:limit]


@pytest.fixture
def settings() -> Settings:
    return Settings(DATABASE_URL="postgresql+asyncpg://test:test@localhost/test")


@pytest.fixture
def sample_stock() -> Stock:
    return Stock(
        symbol="RELIANCE",
        isin="INE002A01018",
        name="Reliance Industries Limited",
        series="EQ",
        sector=None,
        industry=None,
        instrument_type=InstrumentType.EQUITY,
        listing_date=date(1995, 1, 1),
        face_value=Decimal("10.00"),
        is_active=True,
    )


@pytest.fixture
def sample_quote() -> Quote:
    return Quote(
        symbol="RELIANCE",
        last_price=Decimal("2500.00"),
        change=Decimal("12.50"),
        change_percent=Decimal("0.50"),
        open=Decimal("2490.00"),
        high=Decimal("2510.00"),
        low=Decimal("2480.00"),
        previous_close=Decimal("2487.50"),
        volume=1_000_000,
        as_of=datetime.now(UTC),
    )


# --- Shared DB fixtures for tests/integration and tests/api (testcontainers-backed) ---


def _docker_available() -> bool:
    try:
        from testcontainers.postgres import PostgresContainer  # noqa: F401
    except ImportError:
        return False
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


requires_docker = pytest.mark.skipif(not _docker_available(), reason="Docker not available for testcontainers")


@pytest.fixture(scope="session")
def postgres_url() -> str:
    if not _docker_available():
        pytest.skip("Docker not available for testcontainers")

    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16") as container:
        sync_url = container.get_connection_url()
        async_url = sync_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")

        subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=REPO_ROOT,
            env={**os.environ, "DATABASE_URL": async_url},
            check=True,
        )

        yield async_url


@pytest_asyncio.fixture
async def db_session(postgres_url) -> AsyncSession:
    """One AsyncSession per test, against a session-scoped (container-lifetime)
    Postgres. Repository methods commit internally (matches production
    behavior), so a plain rollback-on-teardown does NOT undo committed data -
    tables are truncated up front instead, so each test starts from a clean
    slate regardless of what earlier tests committed or how they ended.
    """
    from sqlalchemy import text

    engine = create_async_engine(postgres_url)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as session:
        await session.execute(
            text(
                "TRUNCATE TABLE financial_results, corporate_actions, historical_prices, stocks "
                "RESTART IDENTITY CASCADE"
            )
        )
        await session.commit()
        yield session
        await session.rollback()
    await engine.dispose()
