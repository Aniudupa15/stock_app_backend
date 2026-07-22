from datetime import date
from decimal import Decimal

from app.core.auth import DEFAULT_USER_ID
from app.domain.entities import PortfolioTransaction, StockMasterRecord, TransactionType
from app.repositories.portfolio_repository import SqlAlchemyPortfolioRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository
from tests.conftest import requires_docker

pytestmark = requires_docker


async def _seed_stock(db_session, symbol: str) -> None:
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [
            StockMasterRecord(
                symbol=symbol, isin=None, name=f"{symbol} Ltd", series="EQ", listing_date=None, face_value=None
            )
        ]
    )
    await db_session.commit()


async def test_create_and_get_portfolio(db_session):
    repo = SqlAlchemyPortfolioRepository(db_session)

    created = await repo.create(DEFAULT_USER_ID, "Long Term")
    fetched = await repo.get(created.id, DEFAULT_USER_ID)

    assert fetched is not None
    assert fetched.name == "Long Term"


async def test_add_transaction_rejects_unknown_symbol(db_session):
    repo = SqlAlchemyPortfolioRepository(db_session)
    portfolio = await repo.create(DEFAULT_USER_ID, "Long Term")

    added = await repo.add_transaction(
        portfolio.id,
        PortfolioTransaction(
            symbol="DOESNOTEXIST",
            transaction_type=TransactionType.BUY,
            quantity=Decimal("10"),
            price=Decimal("100"),
            transaction_date=date(2026, 1, 1),
        ),
    )

    assert added is False


async def test_add_and_get_transactions_ordered_by_date(db_session):
    await _seed_stock(db_session, "RELIANCE")
    repo = SqlAlchemyPortfolioRepository(db_session)
    portfolio = await repo.create(DEFAULT_USER_ID, "Long Term")

    await repo.add_transaction(
        portfolio.id,
        PortfolioTransaction(
            symbol="RELIANCE",
            transaction_type=TransactionType.BUY,
            quantity=Decimal("5"),
            price=Decimal("200"),
            transaction_date=date(2026, 2, 1),
        ),
    )
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

    transactions = await repo.get_transactions(portfolio.id)

    assert [t.transaction_date for t in transactions] == [date(2026, 1, 1), date(2026, 2, 1)]
