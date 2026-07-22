from datetime import UTC, datetime
from decimal import Decimal

from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError
from app.domain.entities import IndexQuote, MarketMover, MarketStatus, NewsArticle, NewsCategory
from app.services.dashboard_service import DashboardService
from app.services.market_mover_service import MarketMoverService
from app.services.news_service import NewsService
from app.services.stock_service import StockService
from tests.conftest import (
    FakeCache,
    FakeMarketMoverRepository,
    FakeNewsProvider,
    FakeNewsRepository,
    FakeStockDataProvider,
    FakeStockRepository,
)


def _mover(symbol: str) -> MarketMover:
    return MarketMover(
        symbol=symbol,
        name=f"{symbol} Ltd",
        last_price=Decimal("100"),
        change=Decimal("1"),
        change_percent=Decimal("1"),
        volume=100,
    )


async def test_dashboard_composes_all_sections():
    settings = Settings(DATABASE_URL="postgresql+asyncpg://test:test@localhost/test")
    stock_provider = FakeStockDataProvider(
        market_statuses=[MarketStatus(market="Capital Market", status="Open", as_of="22-Jul-2026")],
        indices=[
            IndexQuote(
                index_name="NIFTY 50", last_price=Decimal("25000"), change=Decimal("1"), change_percent=Decimal("0.1")
            )
        ],
    )
    stock_service = StockService(FakeStockRepository(), stock_provider, FakeCache(), settings)

    market_mover_repo = FakeMarketMoverRepository(
        top_movers={"gainers": [_mover("UP")], "losers": [_mover("DOWN")]},
        most_active=[_mover("BUSY")],
        extremes={"high": [_mover("NEWHIGH")], "low": [_mover("NEWLOW")]},
    )
    market_mover_service = MarketMoverService(market_mover_repo)

    news_repo = FakeNewsRepository(
        articles=[
            NewsArticle(
                headline="Headline",
                summary=None,
                source="example.com",
                url="https://example.com/1",
                category=NewsCategory.MARKET,
                related_symbols=[],
                published_at=datetime(2026, 7, 22, tzinfo=UTC),
            )
        ]
    )
    news_service = NewsService(FakeNewsProvider(), news_repo, FakeStockRepository())

    service = DashboardService(stock_service, market_mover_service, news_service, FakeCache(), settings)
    dashboard = await service.get_dashboard()

    assert dashboard.market_status[0].market == "Capital Market"
    assert dashboard.indices[0].index_name == "NIFTY 50"
    assert dashboard.gainers[0].symbol == "UP"
    assert dashboard.losers[0].symbol == "DOWN"
    assert dashboard.most_active[0].symbol == "BUSY"
    assert dashboard.fifty_two_week_high[0].symbol == "NEWHIGH"
    assert dashboard.fifty_two_week_low[0].symbol == "NEWLOW"
    assert len(dashboard.latest_news) == 1
    assert len(dashboard.notes) == 3


async def test_dashboard_degrades_gracefully_when_market_status_unavailable():
    settings = Settings(DATABASE_URL="postgresql+asyncpg://test:test@localhost/test")
    stock_provider = FakeStockDataProvider()
    stock_provider.fail_with = ProviderUnavailableError("NSE", "unreachable")
    stock_service = StockService(FakeStockRepository(), stock_provider, FakeCache(), settings)
    market_mover_service = MarketMoverService(FakeMarketMoverRepository())
    news_service = NewsService(FakeNewsProvider(), FakeNewsRepository(), FakeStockRepository())

    service = DashboardService(stock_service, market_mover_service, news_service, FakeCache(), settings)
    dashboard = await service.get_dashboard()

    assert dashboard.market_status == []
    assert dashboard.indices == []
    assert dashboard.gainers == []


async def test_dashboard_response_is_cached_between_calls():
    settings = Settings(DATABASE_URL="postgresql+asyncpg://test:test@localhost/test")
    stock_provider = FakeStockDataProvider(
        market_statuses=[MarketStatus(market="Capital Market", status="Open", as_of="22-Jul-2026")]
    )
    stock_service = StockService(FakeStockRepository(), stock_provider, FakeCache(), settings)
    market_mover_service = MarketMoverService(FakeMarketMoverRepository())
    news_service = NewsService(FakeNewsProvider(), FakeNewsRepository(), FakeStockRepository())
    dashboard_cache = FakeCache()

    service = DashboardService(stock_service, market_mover_service, news_service, dashboard_cache, settings)
    first = await service.get_dashboard()

    # Mutate the underlying source data - if caching works, the second call
    # must still return the first (now-stale) computed result unchanged.
    stock_provider.market_statuses = []
    second = await service.get_dashboard()

    assert second is first
    assert second.market_status[0].market == "Capital Market"
