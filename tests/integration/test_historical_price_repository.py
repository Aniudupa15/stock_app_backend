from datetime import date
from decimal import Decimal

from app.domain.entities import BhavcopyRecord, StockMasterRecord
from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository
from tests.conftest import requires_docker

pytestmark = requires_docker


async def _seed_stock(db_session, symbol: str) -> None:
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [StockMasterRecord(symbol=symbol, isin=None, name=f"{symbol} Ltd", series="EQ", listing_date=None, face_value=None)]
    )
    await db_session.commit()


async def test_bulk_upsert_bars_is_idempotent_and_skips_unknown_symbols(db_session):
    await _seed_stock(db_session, "RELIANCE")
    repo = SqlAlchemyHistoricalPriceRepository(db_session)

    records = [
        BhavcopyRecord(
            symbol="RELIANCE",
            trade_date=date(2026, 7, 17),
            open=Decimal("2490.00"),
            high=Decimal("2510.00"),
            low=Decimal("2480.00"),
            close=Decimal("2500.00"),
            volume=1_000_000,
        ),
        BhavcopyRecord(
            symbol="SGBJUL29",  # not a known equity - should be silently skipped, not error
            trade_date=date(2026, 7, 17),
            open=Decimal("6000.00"),
            high=Decimal("6010.00"),
            low=Decimal("5990.00"),
            close=Decimal("6005.00"),
            volume=100,
        ),
    ]

    first = await repo.bulk_upsert_bars(records)
    second = await repo.bulk_upsert_bars(records)

    assert first == 1  # SGBJUL29 filtered out
    assert second == 1  # re-run updates, doesn't duplicate

    bars = await repo.get_bars("RELIANCE", date(2026, 7, 1), date(2026, 7, 31))
    assert len(bars) == 1
    assert bars[0].close == Decimal("2500.00")


async def test_get_bars_filters_by_date_range_and_orders_ascending(db_session):
    await _seed_stock(db_session, "TCS")
    repo = SqlAlchemyHistoricalPriceRepository(db_session)

    records = [
        BhavcopyRecord(
            symbol="TCS",
            trade_date=d,
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=1000,
        )
        for d in (date(2026, 1, 1), date(2026, 3, 1), date(2026, 6, 1))
    ]
    await repo.bulk_upsert_bars(records)

    bars = await repo.get_bars("TCS", date(2026, 2, 1), date(2026, 12, 31))

    assert [b.trade_date for b in bars] == [date(2026, 3, 1), date(2026, 6, 1)]
