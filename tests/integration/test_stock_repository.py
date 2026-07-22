from datetime import date
from decimal import Decimal

from app.domain.entities import StockMasterRecord
from app.repositories.stock_repository import SqlAlchemyStockRepository
from tests.conftest import requires_docker

pytestmark = requires_docker


async def test_upsert_universe_is_idempotent(db_session):
    repo = SqlAlchemyStockRepository(db_session)
    records = [
        StockMasterRecord(
            symbol="TCS",
            isin="INE467B01029",
            name="Tata Consultancy Services Limited",
            series="EQ",
            listing_date=date(2004, 8, 25),
            face_value=Decimal("1.00"),
        )
    ]

    first = await repo.upsert_universe(records)
    second = await repo.upsert_universe(records)

    assert first == 1
    assert second == 1  # re-running must update, not duplicate

    stock = await repo.get_by_symbol("TCS")
    assert stock is not None
    assert stock.name == "Tata Consultancy Services Limited"


async def test_deactivate_missing_soft_delists(db_session):
    repo = SqlAlchemyStockRepository(db_session)
    await repo.upsert_universe(
        [
            StockMasterRecord(symbol="AAA", isin=None, name="AAA Ltd", series="EQ", listing_date=None, face_value=None),
            StockMasterRecord(symbol="BBB", isin=None, name="BBB Ltd", series="EQ", listing_date=None, face_value=None),
        ]
    )

    deactivated = await repo.deactivate_missing({"AAA"})

    assert deactivated == 1
    assert await repo.get_by_symbol("AAA") is not None
    assert await repo.get_by_symbol("BBB") is None  # is_active filter excludes it


async def test_search_matches_symbol_prefix_and_name_substring(db_session):
    repo = SqlAlchemyStockRepository(db_session)
    await repo.upsert_universe(
        [
            StockMasterRecord(
                symbol="INFY", isin=None, name="Infosys Limited", series="EQ", listing_date=None, face_value=None
            ),
        ]
    )

    by_symbol = await repo.search_by_symbol_or_name("INF", limit=10)
    by_name = await repo.search_by_symbol_or_name("infosys", limit=10)

    assert any(s.symbol == "INFY" for s in by_symbol)
    assert any(s.symbol == "INFY" for s in by_name)


async def test_search_fuzzy_fallback_matches_symbol_and_name_typos(db_session):
    """Regression test: the pg_trgm fuzzy fallback used to only check
    similarity against `name`, so a typo in the *symbol* (no exact prefix
    match, and not a substring of any name) returned nothing even though a
    GIN trigram index exists on `symbol` too.
    """
    repo = SqlAlchemyStockRepository(db_session)
    await repo.upsert_universe(
        [
            StockMasterRecord(
                symbol="RELIANCE",
                isin=None,
                name="Reliance Industries Limited",
                series="EQ",
                listing_date=None,
                face_value=None,
            ),
            StockMasterRecord(
                symbol="INFY", isin=None, name="Infosys Limited", series="EQ", listing_date=None, face_value=None
            ),
        ]
    )

    by_symbol_typo = await repo.search_by_symbol_or_name("RELAINCE", limit=10)  # typo in symbol
    by_name_typo = await repo.search_by_symbol_or_name("Infosis", limit=10)  # typo in name

    assert any(s.symbol == "RELIANCE" for s in by_symbol_typo)
    assert any(s.symbol == "INFY" for s in by_name_typo)
