from datetime import date
from decimal import Decimal

from app.domain.entities import ScreenerFilters, StockIndicatorSnapshot, StockMasterRecord
from app.repositories.screener_repository import SqlAlchemyScreenerRepository
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


def _snapshot(symbol: str, rsi_14=None, close=Decimal("100"), sma_50=None, volume=1000) -> StockIndicatorSnapshot:
    return StockIndicatorSnapshot(
        symbol=symbol,
        name="",
        as_of=date(2026, 7, 22),
        close=close,
        volume=volume,
        rsi_14=rsi_14,
        sma_50=sma_50,
        sma_200=None,
    )


async def test_bulk_upsert_is_idempotent_per_stock(db_session):
    await _seed_stock(db_session, "RELIANCE")
    repo = SqlAlchemyScreenerRepository(db_session)

    first = await repo.bulk_upsert([_snapshot("RELIANCE", rsi_14=Decimal("50"))])
    second = await repo.bulk_upsert([_snapshot("RELIANCE", rsi_14=Decimal("70"))])

    assert first == 1
    assert second == 1

    results = await repo.screen(ScreenerFilters(), limit=10)
    assert len(results) == 1
    assert results[0].rsi_14 == Decimal("70")  # overwritten, not duplicated


async def test_screen_filters_by_rsi_below(db_session):
    await _seed_stock(db_session, "LOWRSI")
    await _seed_stock(db_session, "HIGHRSI")
    repo = SqlAlchemyScreenerRepository(db_session)
    await repo.bulk_upsert([_snapshot("LOWRSI", rsi_14=Decimal("20")), _snapshot("HIGHRSI", rsi_14=Decimal("80"))])

    results = await repo.screen(ScreenerFilters(rsi_below=Decimal("30")), limit=10)

    assert [r.symbol for r in results] == ["LOWRSI"]


async def test_screen_returns_stock_name_via_join(db_session):
    await _seed_stock(db_session, "RELIANCE")
    repo = SqlAlchemyScreenerRepository(db_session)
    await repo.bulk_upsert([_snapshot("RELIANCE")])

    results = await repo.screen(ScreenerFilters(), limit=10)

    assert results[0].name == "RELIANCE Ltd"
