from datetime import date
from decimal import Decimal

from app.domain.entities import FinancialResultRecord, StockMasterRecord
from app.repositories.financial_result_repository import SqlAlchemyFinancialResultRepository
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


async def test_bulk_upsert_is_idempotent_and_skips_unknown_symbols(db_session):
    await _seed_stock(db_session, "RELIANCE")
    repo = SqlAlchemyFinancialResultRepository(db_session)

    records = [
        FinancialResultRecord(
            symbol="RELIANCE",
            period_start=date(2024, 10, 1),
            period_end=date(2024, 12, 31),
            consolidated=False,
            revenue=Decimal("1282600000000.00"),
            profit=Decimal("87210000000.00"),
            eps_basic=Decimal("6.44"),
            eps_diluted=Decimal("6.44"),
        ),
        FinancialResultRecord(
            symbol="NOTAREALSYMBOL",
            period_start=date(2024, 10, 1),
            period_end=date(2024, 12, 31),
            consolidated=False,
            revenue=Decimal("100"),
            profit=Decimal("10"),
            eps_basic=Decimal("1.0"),
            eps_diluted=Decimal("1.0"),
        ),
    ]

    first = await repo.bulk_upsert(records)
    second = await repo.bulk_upsert(records)

    assert first == 1
    assert second == 1

    quarters = await repo.get_recent_quarters("RELIANCE", consolidated=False, limit=5)
    assert len(quarters) == 1
    assert quarters[0].revenue == Decimal("1282600000000.00")


async def test_get_recent_quarters_filters_by_consolidated_flag_and_orders_desc(db_session):
    await _seed_stock(db_session, "TCS")
    repo = SqlAlchemyFinancialResultRepository(db_session)

    await repo.bulk_upsert(
        [
            FinancialResultRecord(
                symbol="TCS",
                period_start=date(2024, 1, 1),
                period_end=date(2024, 3, 31),
                consolidated=False,
                revenue=Decimal("100"),
                profit=Decimal("10"),
                eps_basic=Decimal("1.0"),
                eps_diluted=Decimal("1.0"),
            ),
            FinancialResultRecord(
                symbol="TCS",
                period_start=date(2024, 4, 1),
                period_end=date(2024, 6, 30),
                consolidated=False,
                revenue=Decimal("110"),
                profit=Decimal("11"),
                eps_basic=Decimal("1.1"),
                eps_diluted=Decimal("1.1"),
            ),
            FinancialResultRecord(
                symbol="TCS",
                period_start=date(2024, 4, 1),
                period_end=date(2024, 6, 30),
                consolidated=True,
                revenue=Decimal("500"),
                profit=Decimal("50"),
                eps_basic=Decimal("5.0"),
                eps_diluted=Decimal("5.0"),
            ),
        ]
    )

    standalone = await repo.get_recent_quarters("TCS", consolidated=False, limit=10)
    consolidated = await repo.get_recent_quarters("TCS", consolidated=True, limit=10)

    assert [q.period_end for q in standalone] == [date(2024, 6, 30), date(2024, 3, 31)]
    assert len(consolidated) == 1
    assert consolidated[0].revenue == Decimal("500")
