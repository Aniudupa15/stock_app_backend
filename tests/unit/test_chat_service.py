import uuid
from datetime import date, timedelta
from decimal import Decimal

import pytest

from app.domain.entities import MarketMover, OhlcvBar, PortfolioTransaction, TransactionType
from app.schemas.chat import ChatIntent
from app.services.chat_service import ChatService
from app.services.indicator_service import IndicatorService
from app.services.notification_service import NotificationService
from app.services.portfolio_service import PortfolioService
from app.services.stock_service import StockService
from app.services.watchlist_service import WatchlistService
from tests.conftest import (
    FakeCache,
    FakeHistoricalPriceRepository,
    FakeMarketMoverRepository,
    FakeNotificationRepository,
    FakePortfolioRepository,
    FakeStockDataProvider,
    FakeStockRepository,
    FakeWatchlistRepository,
)

USER_ID = uuid.uuid4()


def _bars(n: int) -> list[OhlcvBar]:
    start = date(2026, 1, 1)
    return [
        OhlcvBar(
            trade_date=start + timedelta(days=i),
            open=Decimal("100") + i,
            high=Decimal("101") + i,
            low=Decimal("99") + i,
            close=Decimal("100") + i,
            volume=1000,
        )
        for i in range(n)
    ]


@pytest.fixture
def service(sample_stock, sample_quote, settings) -> ChatService:
    stock_repo = FakeStockRepository([sample_stock])
    stock_service = StockService(
        stock_repo, FakeStockDataProvider(quotes={"RELIANCE": sample_quote}), FakeCache(), settings
    )
    indicator_service = IndicatorService(stock_repo, FakeHistoricalPriceRepository({"RELIANCE": _bars(60)}))
    market_mover_repo = FakeMarketMoverRepository()
    portfolio_service = PortfolioService(FakePortfolioRepository({"RELIANCE"}), market_mover_repo)
    watchlist_service = WatchlistService(FakeWatchlistRepository({"RELIANCE"}), market_mover_repo)
    notification_service = NotificationService(FakeNotificationRepository())
    return ChatService(
        stock_repo, stock_service, indicator_service, portfolio_service, watchlist_service, notification_service
    )


async def test_unknown_message_returns_help_text(service):
    result = await service.ask(USER_ID, "what's the weather today")

    assert result.intent == ChatIntent.UNKNOWN


async def test_indicator_question_matches_known_symbol(service):
    result = await service.ask(USER_ID, "what's the RSI for RELIANCE")

    assert result.intent == ChatIntent.INDICATOR_SUMMARY
    assert "RELIANCE" in result.answer
    assert "RSI(14)" in result.answer


async def test_quote_question_matches_known_symbol(service):
    result = await service.ask(USER_ID, "what's the price of RELIANCE")

    assert result.intent == ChatIntent.STOCK_QUOTE
    assert "RELIANCE" in result.answer


async def test_portfolio_question_with_no_portfolios(service):
    result = await service.ask(USER_ID, "what's my portfolio doing")

    assert result.intent == ChatIntent.PORTFOLIO_SUMMARY
    assert "don't have any portfolios" in result.answer


async def test_portfolio_question_with_a_holding(service):
    repo = FakePortfolioRepository({"RELIANCE"})
    market_mover_repo = FakeMarketMoverRepository(
        latest_prices={
            "RELIANCE": MarketMover(
                symbol="RELIANCE",
                name="Reliance Industries Limited",
                last_price=Decimal("120"),
                change=Decimal("10"),
                change_percent=Decimal("9.09"),
                volume=1000,
            )
        }
    )
    portfolio = await repo.create(USER_ID, "Main")
    await repo.add_transaction(
        portfolio.id,
        PortfolioTransaction(
            symbol="RELIANCE",
            transaction_type=TransactionType.BUY,
            quantity=Decimal("10"),
            price=Decimal("100"),
            transaction_date=date(2026, 1, 1),
        ),
    )
    service._portfolio_service = PortfolioService(repo, market_mover_repo)

    result = await service.ask(USER_ID, "how is my portfolio doing")

    assert result.intent == ChatIntent.PORTFOLIO_SUMMARY
    assert "Main" in result.answer


async def test_watchlist_question_with_no_watchlists(service):
    result = await service.ask(USER_ID, "what's in my watchlist")

    assert result.intent == ChatIntent.WATCHLIST_SUMMARY
    assert "don't have any watchlists" in result.answer


async def test_alerts_question_with_no_notifications(service):
    result = await service.ask(USER_ID, "any new alerts")

    assert result.intent == ChatIntent.ALERTS_SUMMARY
    assert "no unread notifications" in result.answer


async def test_alerts_question_with_unread_notification(service):
    repo = FakeNotificationRepository()
    await repo.create(USER_ID, None, "Alert triggered", "RELIANCE crossed 3000")
    service._notification_service = NotificationService(repo)

    result = await service.ask(USER_ID, "do I have any notifications")

    assert result.intent == ChatIntent.ALERTS_SUMMARY
    assert "Alert triggered" in result.answer


async def test_stopwords_alone_do_not_match_a_symbol(service):
    result = await service.ask(USER_ID, "what is my rsi doing today")

    # "RSI" itself isn't a known stock symbol and no other token matches -
    # falls through to unknown rather than crashing on a bogus lookup.
    assert result.intent == ChatIntent.UNKNOWN
