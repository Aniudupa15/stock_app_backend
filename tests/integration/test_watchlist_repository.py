import uuid

from app.core.auth import DEFAULT_USER_ID
from app.domain.entities import StockMasterRecord
from app.repositories.stock_repository import SqlAlchemyStockRepository
from app.repositories.watchlist_repository import SqlAlchemyWatchlistRepository
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


async def test_create_list_and_get_watchlist(db_session):
    repo = SqlAlchemyWatchlistRepository(db_session)

    created = await repo.create(DEFAULT_USER_ID, "My Picks")
    assert created.name == "My Picks"

    listed = await repo.list_for_user(DEFAULT_USER_ID)
    assert [w.id for w in listed] == [created.id]

    fetched = await repo.get(created.id, DEFAULT_USER_ID)
    assert fetched is not None
    assert fetched.name == "My Picks"


async def test_get_returns_none_for_wrong_user(db_session):
    repo = SqlAlchemyWatchlistRepository(db_session)
    created = await repo.create(DEFAULT_USER_ID, "My Picks")

    result = await repo.get(created.id, uuid.uuid4())

    assert result is None


async def test_add_item_is_idempotent_and_rejects_unknown_symbol(db_session):
    await _seed_stock(db_session, "RELIANCE")
    repo = SqlAlchemyWatchlistRepository(db_session)
    watchlist = await repo.create(DEFAULT_USER_ID, "My Picks")

    first = await repo.add_item(watchlist.id, "RELIANCE")
    second = await repo.add_item(watchlist.id, "reliance")
    unknown = await repo.add_item(watchlist.id, "DOESNOTEXIST")

    assert first is True
    assert second is True
    assert unknown is False

    items = await repo.get_items(watchlist.id)
    assert len(items) == 1
    assert items[0].symbol == "RELIANCE"


async def test_remove_item(db_session):
    await _seed_stock(db_session, "TCS")
    repo = SqlAlchemyWatchlistRepository(db_session)
    watchlist = await repo.create(DEFAULT_USER_ID, "My Picks")
    await repo.add_item(watchlist.id, "TCS")

    removed = await repo.remove_item(watchlist.id, "TCS")
    removed_again = await repo.remove_item(watchlist.id, "TCS")

    assert removed is True
    assert removed_again is False
    assert await repo.get_items(watchlist.id) == []


async def test_delete_watchlist_cascades_items(db_session):
    await _seed_stock(db_session, "INFY")
    repo = SqlAlchemyWatchlistRepository(db_session)
    watchlist = await repo.create(DEFAULT_USER_ID, "My Picks")
    await repo.add_item(watchlist.id, "INFY")

    deleted = await repo.delete(watchlist.id, DEFAULT_USER_ID)
    deleted_again = await repo.delete(watchlist.id, DEFAULT_USER_ID)

    assert deleted is True
    assert deleted_again is False
    assert await repo.get(watchlist.id, DEFAULT_USER_ID) is None
