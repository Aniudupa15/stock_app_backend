import uuid

import pytest

from app.core.exceptions import StockNotFoundError, WatchlistNotFoundError
from app.domain.entities import MarketMover
from app.services.watchlist_service import WatchlistService
from tests.conftest import FakeMarketMoverRepository, FakeWatchlistRepository

USER_ID = uuid.uuid4()
OTHER_USER_ID = uuid.uuid4()


async def test_create_and_list_watchlists():
    repo = FakeWatchlistRepository()
    service = WatchlistService(repo, FakeMarketMoverRepository())

    created = await service.create(USER_ID, "My Picks")
    assert created.name == "My Picks"
    assert created.item_count == 0

    listed = await service.list(USER_ID)
    assert len(listed) == 1
    assert listed[0].id == created.id


async def test_list_only_returns_own_watchlists():
    repo = FakeWatchlistRepository()
    service = WatchlistService(repo, FakeMarketMoverRepository())

    await service.create(USER_ID, "Mine")
    await service.create(OTHER_USER_ID, "Theirs")

    listed = await service.list(USER_ID)
    assert [w.name for w in listed] == ["Mine"]


async def test_get_detail_unknown_watchlist_raises():
    repo = FakeWatchlistRepository()
    service = WatchlistService(repo, FakeMarketMoverRepository())

    with pytest.raises(WatchlistNotFoundError):
        await service.get_detail(USER_ID, uuid.uuid4())


async def test_get_detail_for_other_users_watchlist_raises():
    repo = FakeWatchlistRepository()
    service = WatchlistService(repo, FakeMarketMoverRepository())
    watchlist = await service.create(OTHER_USER_ID, "Theirs")

    with pytest.raises(WatchlistNotFoundError):
        await service.get_detail(USER_ID, watchlist.id)


async def test_add_symbol_unknown_symbol_raises_stock_not_found():
    repo = FakeWatchlistRepository(known_symbols=set())
    service = WatchlistService(repo, FakeMarketMoverRepository())
    watchlist = await service.create(USER_ID, "My Picks")

    with pytest.raises(StockNotFoundError):
        await service.add_symbol(USER_ID, watchlist.id, "DOESNOTEXIST")


async def test_add_symbol_composes_latest_price_into_detail():
    from decimal import Decimal

    repo = FakeWatchlistRepository(known_symbols={"RELIANCE"})
    price_repo = FakeMarketMoverRepository(
        latest_prices={
            "RELIANCE": MarketMover(
                symbol="RELIANCE",
                name="Reliance Ltd",
                last_price=Decimal("2500"),
                change=Decimal("10"),
                change_percent=Decimal("0.4"),
                volume=100,
            )
        }
    )
    service = WatchlistService(repo, price_repo)
    watchlist = await service.create(USER_ID, "My Picks")

    detail = await service.add_symbol(USER_ID, watchlist.id, "reliance")

    assert len(detail.items) == 1
    assert detail.items[0].symbol == "RELIANCE"
    assert detail.items[0].last_price == Decimal("2500")


async def test_remove_symbol_is_idempotent_for_missing_item():
    repo = FakeWatchlistRepository(known_symbols={"RELIANCE"})
    service = WatchlistService(repo, FakeMarketMoverRepository())
    watchlist = await service.create(USER_ID, "My Picks")

    detail = await service.remove_symbol(USER_ID, watchlist.id, "RELIANCE")

    assert detail.items == []


async def test_delete_unknown_watchlist_raises():
    repo = FakeWatchlistRepository()
    service = WatchlistService(repo, FakeMarketMoverRepository())

    with pytest.raises(WatchlistNotFoundError):
        await service.delete(USER_ID, uuid.uuid4())
