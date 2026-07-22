from datetime import date

from app.domain.entities import CorporateAction, StockMasterRecord
from app.repositories.corporate_action_repository import SqlAlchemyCorporateActionRepository
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
    repo = SqlAlchemyCorporateActionRepository(db_session)

    records = [
        CorporateAction(
            symbol="RELIANCE",
            purpose="Dividend - Rs 10 Per Share",
            face_value=None,
            ex_date=date(2026, 8, 1),
            record_date=date(2026, 8, 1),
            book_closure_start=None,
            book_closure_end=None,
        ),
        CorporateAction(
            symbol="NOTAREALSYMBOL",
            purpose="Dividend",
            face_value=None,
            ex_date=date(2026, 8, 1),
            record_date=None,
            book_closure_start=None,
            book_closure_end=None,
        ),
    ]

    first = await repo.bulk_upsert(records)
    second = await repo.bulk_upsert(records)

    assert first == 1
    assert second == 1

    actions = await repo.get_for_symbol("RELIANCE")
    assert len(actions) == 1
    assert actions[0].purpose == "Dividend - Rs 10 Per Share"


async def test_get_for_symbol_returns_empty_for_unknown_symbol(db_session):
    repo = SqlAlchemyCorporateActionRepository(db_session)
    assert await repo.get_for_symbol("DOESNOTEXIST") == []
