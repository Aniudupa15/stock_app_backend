import uuid

from app.core.exceptions import StockNotFoundError, WatchlistNotFoundError
from app.domain.ports import MarketMoverRepositoryPort, WatchlistRepositoryPort
from app.schemas.watchlist import WatchlistDetailOut, WatchlistItemOut, WatchlistOut


class WatchlistService:
    def __init__(self, repository: WatchlistRepositoryPort, price_repository: MarketMoverRepositoryPort):
        self._repository = repository
        self._price_repository = price_repository

    async def create(self, user_id: uuid.UUID, name: str) -> WatchlistOut:
        watchlist = await self._repository.create(user_id, name)
        return WatchlistOut(id=watchlist.id, name=watchlist.name, created_at=watchlist.created_at, item_count=0)

    async def list(self, user_id: uuid.UUID) -> list[WatchlistOut]:
        watchlists = await self._repository.list_for_user(user_id)
        result = []
        for w in watchlists:
            items = await self._repository.get_items(w.id)
            result.append(WatchlistOut(id=w.id, name=w.name, created_at=w.created_at, item_count=len(items)))
        return result

    async def get_detail(self, user_id: uuid.UUID, watchlist_id: uuid.UUID) -> WatchlistDetailOut:
        watchlist = await self._repository.get(watchlist_id, user_id)
        if watchlist is None:
            raise WatchlistNotFoundError(watchlist_id)

        items = await self._repository.get_items(watchlist_id)
        prices = await self._price_repository.get_latest_prices([i.symbol for i in items])
        price_by_symbol = {p.symbol: p for p in prices}

        item_outs = [
            WatchlistItemOut(
                symbol=i.symbol,
                name=i.name,
                added_at=i.added_at,
                last_price=price_by_symbol[i.symbol].last_price if i.symbol in price_by_symbol else None,
                change=price_by_symbol[i.symbol].change if i.symbol in price_by_symbol else None,
                change_percent=price_by_symbol[i.symbol].change_percent if i.symbol in price_by_symbol else None,
            )
            for i in items
        ]
        return WatchlistDetailOut(
            id=watchlist.id, name=watchlist.name, created_at=watchlist.created_at, items=item_outs
        )

    async def delete(self, user_id: uuid.UUID, watchlist_id: uuid.UUID) -> None:
        deleted = await self._repository.delete(watchlist_id, user_id)
        if not deleted:
            raise WatchlistNotFoundError(watchlist_id)

    async def add_symbol(self, user_id: uuid.UUID, watchlist_id: uuid.UUID, symbol: str) -> WatchlistDetailOut:
        watchlist = await self._repository.get(watchlist_id, user_id)
        if watchlist is None:
            raise WatchlistNotFoundError(watchlist_id)

        added = await self._repository.add_item(watchlist_id, symbol)
        if not added:
            raise StockNotFoundError(symbol)

        return await self.get_detail(user_id, watchlist_id)

    async def remove_symbol(self, user_id: uuid.UUID, watchlist_id: uuid.UUID, symbol: str) -> WatchlistDetailOut:
        watchlist = await self._repository.get(watchlist_id, user_id)
        if watchlist is None:
            raise WatchlistNotFoundError(watchlist_id)

        await self._repository.remove_item(watchlist_id, symbol)
        return await self.get_detail(user_id, watchlist_id)
