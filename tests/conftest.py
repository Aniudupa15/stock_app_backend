import os
import subprocess
import sys
import uuid
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.domain.entities import (
    Alert,
    AlertStatus,
    AlertType,
    BhavcopyRecord,
    CorporateAction,
    FinancialResultFiling,
    FinancialResultRecord,
    IndexQuote,
    InstrumentType,
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
    Stock,
    StockMasterRecord,
    User,
    Watchlist,
    WatchlistItem,
)
from app.domain.ports import (
    AlertRepositoryPort,
    CachePort,
    CorporateActionRepositoryPort,
    FinancialResultRepositoryPort,
    HistoricalPriceRepositoryPort,
    MarketMoverRepositoryPort,
    NewsProviderPort,
    NewsRepositoryPort,
    NotificationRepositoryPort,
    PortfolioRepositoryPort,
    RefreshTokenRepositoryPort,
    StockDataProviderPort,
    StockRepositoryPort,
    UserRepositoryPort,
    WatchlistRepositoryPort,
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
        matches = [s for s in self.stocks.values() if q in s.symbol.lower() or q in s.name.lower()]
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

    async def list_active_symbols(self) -> list[str]:
        return [s.symbol for s in self.stocks.values() if s.is_active]


class FakeStockDataProvider(StockDataProviderPort):
    def __init__(
        self,
        quotes: dict[str, Quote] | None = None,
        universe: list[StockMasterRecord] | None = None,
        daily_bars: list[BhavcopyRecord] | None = None,
        corporate_actions: list[CorporateAction] | None = None,
        financial_filings: list[FinancialResultFiling] | None = None,
        financial_details: dict[str, FinancialResultRecord] | None = None,
        market_statuses: list[MarketStatus] | None = None,
        indices: list[IndexQuote] | None = None,
    ):
        self.quotes = quotes or {}
        self.universe = universe or []
        self.daily_bars = daily_bars or []
        self.corporate_actions = corporate_actions or []
        self.financial_filings = financial_filings or []
        self.financial_details = financial_details or {}  # keyed by xbrl_url
        self.market_statuses = market_statuses or []
        self.indices = indices or []
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

    async def fetch_market_status(self) -> list[MarketStatus]:
        if self.fail_with:
            raise self.fail_with
        return self.market_statuses

    async def fetch_indices(self) -> list[IndexQuote]:
        if self.fail_with:
            raise self.fail_with
        return self.indices


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


class FakeMarketMoverRepository(MarketMoverRepositoryPort):
    def __init__(
        self,
        top_movers: dict[str, list[MarketMover]] | None = None,
        most_active: list[MarketMover] | None = None,
        extremes: dict[str, list[MarketMover]] | None = None,
        latest_prices: dict[str, MarketMover] | None = None,
    ):
        self.top_movers = top_movers or {}  # keyed by "gainers"/"losers"
        self.most_active = most_active or []
        self.extremes = extremes or {}  # keyed by "high"/"low"
        self.latest_prices = latest_prices or {}  # keyed by symbol
        self.calls: list[tuple] = []

    async def get_top_movers(self, direction: str, lookback_sessions: int, limit: int) -> list[MarketMover]:
        self.calls.append(("get_top_movers", direction, lookback_sessions, limit))
        return self.top_movers.get(direction, [])[:limit]

    async def get_most_active(self, limit: int) -> list[MarketMover]:
        self.calls.append(("get_most_active", limit))
        return self.most_active[:limit]

    async def get_latest_prices(self, symbols: list[str]) -> list[MarketMover]:
        self.calls.append(("get_latest_prices", tuple(symbols)))
        return [self.latest_prices[s] for s in symbols if s in self.latest_prices]

    async def get_52_week_extremes(self, direction: str, limit: int) -> list[MarketMover]:
        self.calls.append(("get_52_week_extremes", direction, limit))
        return self.extremes.get(direction, [])[:limit]


class FakeWatchlistRepository(WatchlistRepositoryPort):
    def __init__(self, known_symbols: set[str] | None = None):
        self.known_symbols = known_symbols or set()
        self.watchlists: dict[uuid.UUID, Watchlist] = {}
        self.items: dict[uuid.UUID, list[WatchlistItem]] = {}

    async def create(self, user_id: uuid.UUID, name: str) -> Watchlist:
        watchlist = Watchlist(id=uuid.uuid4(), user_id=user_id, name=name, created_at=datetime.now(UTC))
        self.watchlists[watchlist.id] = watchlist
        self.items[watchlist.id] = []
        return watchlist

    async def list_for_user(self, user_id: uuid.UUID) -> list[Watchlist]:
        return [w for w in self.watchlists.values() if w.user_id == user_id]

    async def get(self, watchlist_id: uuid.UUID, user_id: uuid.UUID) -> Watchlist | None:
        w = self.watchlists.get(watchlist_id)
        if w is None or w.user_id != user_id:
            return None
        return w

    async def delete(self, watchlist_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        w = self.watchlists.get(watchlist_id)
        if w is None or w.user_id != user_id:
            return False
        del self.watchlists[watchlist_id]
        self.items.pop(watchlist_id, None)
        return True

    async def add_item(self, watchlist_id: uuid.UUID, symbol: str) -> bool:
        symbol = symbol.strip().upper()
        if symbol not in self.known_symbols:
            return False
        existing = self.items.setdefault(watchlist_id, [])
        if not any(i.symbol == symbol for i in existing):
            existing.append(WatchlistItem(symbol=symbol, name=f"{symbol} Ltd", added_at=datetime.now(UTC)))
        return True

    async def remove_item(self, watchlist_id: uuid.UUID, symbol: str) -> bool:
        symbol = symbol.strip().upper()
        existing = self.items.get(watchlist_id, [])
        before = len(existing)
        self.items[watchlist_id] = [i for i in existing if i.symbol != symbol]
        return len(self.items[watchlist_id]) != before

    async def get_items(self, watchlist_id: uuid.UUID) -> list[WatchlistItem]:
        return self.items.get(watchlist_id, [])


class FakePortfolioRepository(PortfolioRepositoryPort):
    def __init__(self, known_symbols: set[str] | None = None):
        self.known_symbols = known_symbols or set()
        self.portfolios: dict[uuid.UUID, Portfolio] = {}
        self.transactions: dict[uuid.UUID, list[PortfolioTransaction]] = {}

    async def create(self, user_id: uuid.UUID, name: str) -> Portfolio:
        portfolio = Portfolio(id=uuid.uuid4(), user_id=user_id, name=name, created_at=datetime.now(UTC))
        self.portfolios[portfolio.id] = portfolio
        self.transactions[portfolio.id] = []
        return portfolio

    async def list_for_user(self, user_id: uuid.UUID) -> list[Portfolio]:
        return [p for p in self.portfolios.values() if p.user_id == user_id]

    async def get(self, portfolio_id: uuid.UUID, user_id: uuid.UUID) -> Portfolio | None:
        p = self.portfolios.get(portfolio_id)
        if p is None or p.user_id != user_id:
            return None
        return p

    async def add_transaction(self, portfolio_id: uuid.UUID, transaction: PortfolioTransaction) -> bool:
        if transaction.symbol not in self.known_symbols:
            return False
        self.transactions.setdefault(portfolio_id, []).append(transaction)
        return True

    async def get_transactions(self, portfolio_id: uuid.UUID) -> list[PortfolioTransaction]:
        return sorted(self.transactions.get(portfolio_id, []), key=lambda t: t.transaction_date)


class FakeNewsProvider(NewsProviderPort):
    def __init__(self, articles: list[NewsArticle] | None = None):
        self.articles = articles or []
        self.fail_with: Exception | None = None
        self.last_known_symbols: set[str] | None = None

    async def fetch_latest(self, known_symbols: set[str]) -> list[NewsArticle]:
        self.last_known_symbols = known_symbols
        if self.fail_with:
            raise self.fail_with
        return self.articles


class FakeNewsRepository(NewsRepositoryPort):
    def __init__(self, articles: list[NewsArticle] | None = None):
        self.articles_by_url: dict[str, NewsArticle] = {a.url: a for a in (articles or [])}

    async def bulk_upsert(self, articles: list[NewsArticle]) -> int:
        for a in articles:
            self.articles_by_url[a.url] = a
        return len(articles)

    async def list_latest(
        self, category: NewsCategory | None, symbol: str | None, limit: int, offset: int
    ) -> list[NewsArticle]:
        articles = sorted(self.articles_by_url.values(), key=lambda a: a.published_at, reverse=True)
        if category is not None:
            articles = [a for a in articles if a.category == category]
        if symbol is not None:
            articles = [a for a in articles if symbol.strip().upper() in a.related_symbols]
        return articles[offset : offset + limit]


class FakeAlertRepository(AlertRepositoryPort):
    def __init__(self, known_symbols: set[str] | None = None):
        self.known_symbols = known_symbols or set()
        self.alerts: dict[uuid.UUID, Alert] = {}

    async def create(self, user_id: uuid.UUID, symbol: str, alert_type: AlertType, condition: dict) -> Alert | None:
        if symbol not in self.known_symbols:
            return None
        alert = Alert(
            id=uuid.uuid4(),
            user_id=user_id,
            symbol=symbol,
            alert_type=alert_type,
            condition=condition,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(UTC),
            triggered_at=None,
        )
        self.alerts[alert.id] = alert
        return alert

    async def list_for_user(self, user_id: uuid.UUID, status: AlertStatus | None) -> list[Alert]:
        return [a for a in self.alerts.values() if a.user_id == user_id and (status is None or a.status == status)]

    async def get(self, alert_id: uuid.UUID, user_id: uuid.UUID) -> Alert | None:
        a = self.alerts.get(alert_id)
        if a is None or a.user_id != user_id:
            return None
        return a

    async def delete(self, alert_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        a = self.alerts.get(alert_id)
        if a is None or a.user_id != user_id:
            return False
        del self.alerts[alert_id]
        return True

    async def list_active(self) -> list[Alert]:
        return [a for a in self.alerts.values() if a.status == AlertStatus.ACTIVE]

    async def mark_triggered(self, alert_id: uuid.UUID, triggered_at: datetime) -> None:
        a = self.alerts[alert_id]
        self.alerts[alert_id] = replace(a, status=AlertStatus.TRIGGERED, triggered_at=triggered_at)


class FakeNotificationRepository(NotificationRepositoryPort):
    def __init__(self):
        self.notifications: dict[uuid.UUID, Notification] = {}

    async def create(self, user_id: uuid.UUID, alert_id: uuid.UUID | None, title: str, message: str) -> Notification:
        notification = Notification(
            id=uuid.uuid4(),
            user_id=user_id,
            alert_id=alert_id,
            title=title,
            message=message,
            created_at=datetime.now(UTC),
            read_at=None,
        )
        self.notifications[notification.id] = notification
        return notification

    async def list_for_user(self, user_id: uuid.UUID, unread_only: bool, limit: int, offset: int) -> list[Notification]:
        items = [
            n for n in self.notifications.values() if n.user_id == user_id and (not unread_only or n.read_at is None)
        ]
        items.sort(key=lambda n: n.created_at, reverse=True)
        return items[offset : offset + limit]

    async def mark_read(self, notification_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        n = self.notifications.get(notification_id)
        if n is None or n.user_id != user_id:
            return False
        self.notifications[notification_id] = replace(n, read_at=datetime.now(UTC))
        return True


class FakeUserRepository(UserRepositoryPort):
    def __init__(self, users: list[User] | None = None):
        self.users_by_id: dict[uuid.UUID, User] = {u.id: u for u in (users or [])}

    async def get_by_email(self, email: str) -> User | None:
        email = email.strip().lower()
        return next((u for u in self.users_by_id.values() if u.email == email), None)

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        return self.users_by_id.get(user_id)

    async def create(self, email: str, password_hash: str, display_name: str) -> User:
        user = User(
            id=uuid.uuid4(),
            email=email.strip().lower(),
            display_name=display_name,
            password_hash=password_hash,
            created_at=datetime.now(UTC),
        )
        self.users_by_id[user.id] = user
        return user


class FakeRefreshTokenRepository(RefreshTokenRepositoryPort):
    def __init__(self):
        self.tokens_by_id: dict[uuid.UUID, RefreshToken] = {}

    async def create(self, user_id: uuid.UUID, token_hash: str, expires_at: datetime) -> RefreshToken:
        token = RefreshToken(
            id=uuid.uuid4(),
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
            revoked_at=None,
            created_at=datetime.now(UTC),
        )
        self.tokens_by_id[token.id] = token
        return token

    async def get_by_hash(self, token_hash: str) -> RefreshToken | None:
        return next((t for t in self.tokens_by_id.values() if t.token_hash == token_hash), None)

    async def revoke(self, token_id: uuid.UUID) -> None:
        token = self.tokens_by_id.get(token_id)
        if token is not None:
            self.tokens_by_id[token_id] = replace(token, revoked_at=datetime.now(UTC))


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
                "TRUNCATE TABLE notifications, alerts, news_articles, portfolio_transactions, portfolios, "
                "watchlist_items, watchlists, financial_results, corporate_actions, historical_prices, stocks "
                "RESTART IDENTITY CASCADE"
            )
        )
        await session.commit()
        yield session
        await session.rollback()
    await engine.dispose()
