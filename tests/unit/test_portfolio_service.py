import uuid
from datetime import date
from decimal import Decimal

import pytest

from app.core.exceptions import PortfolioNotFoundError, StockNotFoundError
from app.domain.entities import MarketMover
from app.schemas.portfolio import TransactionCreate
from app.services.portfolio_service import PortfolioService
from tests.conftest import FakeMarketMoverRepository, FakePortfolioRepository

USER_ID = uuid.uuid4()
OTHER_USER_ID = uuid.uuid4()


def _reliance_price(price: str) -> MarketMover:
    return MarketMover(
        symbol="RELIANCE",
        name="Reliance Ltd",
        last_price=Decimal(price),
        change=Decimal("0"),
        change_percent=Decimal("0"),
        volume=100,
    )


async def test_create_and_list_portfolios():
    repo = FakePortfolioRepository()
    service = PortfolioService(repo, FakeMarketMoverRepository())

    created = await service.create(USER_ID, "Long Term")
    listed = await service.list(USER_ID)

    assert listed[0].id == created.id


async def test_add_transaction_unknown_symbol_raises():
    repo = FakePortfolioRepository(known_symbols=set())
    service = PortfolioService(repo, FakeMarketMoverRepository())
    portfolio = await service.create(USER_ID, "Long Term")

    with pytest.raises(StockNotFoundError):
        await service.add_transaction(
            USER_ID,
            portfolio.id,
            TransactionCreate(
                symbol="DOESNOTEXIST",
                transaction_type="BUY",
                quantity=Decimal("10"),
                price=Decimal("100"),
                transaction_date=date(2026, 1, 1),
            ),
        )


async def test_add_transaction_unknown_portfolio_raises():
    repo = FakePortfolioRepository(known_symbols={"RELIANCE"})
    service = PortfolioService(repo, FakeMarketMoverRepository())

    with pytest.raises(PortfolioNotFoundError):
        await service.add_transaction(
            USER_ID,
            uuid.uuid4(),
            TransactionCreate(
                symbol="RELIANCE",
                transaction_type="BUY",
                quantity=Decimal("10"),
                price=Decimal("100"),
                transaction_date=date(2026, 1, 1),
            ),
        )


async def test_get_detail_computes_holdings_with_current_price():
    repo = FakePortfolioRepository(known_symbols={"RELIANCE"})
    price_repo = FakeMarketMoverRepository(latest_prices={"RELIANCE": _reliance_price("150")})
    service = PortfolioService(repo, price_repo)
    portfolio = await service.create(USER_ID, "Long Term")

    detail = await service.add_transaction(
        USER_ID,
        portfolio.id,
        TransactionCreate(
            symbol="RELIANCE",
            transaction_type="BUY",
            quantity=Decimal("10"),
            price=Decimal("100"),
            transaction_date=date(2026, 1, 1),
        ),
    )

    assert len(detail.holdings) == 1
    holding = detail.holdings[0]
    assert holding.quantity == Decimal("10")
    assert holding.avg_price == Decimal("100")
    assert holding.current_price == Decimal("150")
    assert holding.current_value == Decimal("1500")
    assert holding.pnl == Decimal("500")


async def test_get_detail_other_users_portfolio_raises():
    repo = FakePortfolioRepository()
    service = PortfolioService(repo, FakeMarketMoverRepository())
    portfolio = await service.create(OTHER_USER_ID, "Theirs")

    with pytest.raises(PortfolioNotFoundError):
        await service.get_detail(USER_ID, portfolio.id)


async def test_get_performance_computes_pnl_and_xirr():
    repo = FakePortfolioRepository(known_symbols={"RELIANCE"})
    price_repo = FakeMarketMoverRepository(latest_prices={"RELIANCE": _reliance_price("110")})
    service = PortfolioService(repo, price_repo)
    portfolio = await service.create(USER_ID, "Long Term")
    await service.add_transaction(
        USER_ID,
        portfolio.id,
        TransactionCreate(
            symbol="RELIANCE",
            transaction_type="BUY",
            quantity=Decimal("10"),
            price=Decimal("100"),
            transaction_date=date(2025, 1, 1),
        ),
    )

    performance = await service.get_performance(USER_ID, portfolio.id)

    assert performance.total_invested == Decimal("1000")
    assert performance.current_value == Decimal("1100")
    assert performance.total_pnl == Decimal("100")
    assert performance.xirr_percent is not None
