import uuid

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import Watchlist, WatchlistItem
from app.domain.ports import WatchlistRepositoryPort
from app.models.stock import StockModel
from app.models.watchlist import WatchlistModel
from app.models.watchlist_item import WatchlistItemModel


class SqlAlchemyWatchlistRepository(WatchlistRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, user_id: uuid.UUID, name: str) -> Watchlist:
        model = WatchlistModel(user_id=user_id, name=name)
        self._session.add(model)
        await self._session.commit()
        await self._session.refresh(model)
        return Watchlist(id=model.id, user_id=model.user_id, name=model.name, created_at=model.created_at)

    async def list_for_user(self, user_id: uuid.UUID) -> list[Watchlist]:
        stmt = select(WatchlistModel).where(WatchlistModel.user_id == user_id).order_by(WatchlistModel.created_at.asc())
        result = await self._session.execute(stmt)
        return [
            Watchlist(id=row.id, user_id=row.user_id, name=row.name, created_at=row.created_at)
            for row in result.scalars().all()
        ]

    async def get(self, watchlist_id: uuid.UUID, user_id: uuid.UUID) -> Watchlist | None:
        stmt = select(WatchlistModel).where(WatchlistModel.id == watchlist_id, WatchlistModel.user_id == user_id)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return Watchlist(id=row.id, user_id=row.user_id, name=row.name, created_at=row.created_at)

    async def delete(self, watchlist_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        stmt = (
            delete(WatchlistModel)
            .where(WatchlistModel.id == watchlist_id, WatchlistModel.user_id == user_id)
            .returning(WatchlistModel.id)
        )
        result = await self._session.execute(stmt)
        deleted = result.first() is not None
        await self._session.commit()
        return deleted

    async def add_item(self, watchlist_id: uuid.UUID, symbol: str) -> bool:
        stock_id = await self._resolve_stock_id(symbol)
        if stock_id is None:
            return False

        stmt = (
            pg_insert(WatchlistItemModel)
            .values(watchlist_id=watchlist_id, stock_id=stock_id)
            .on_conflict_do_nothing(index_elements=["watchlist_id", "stock_id"])
        )
        await self._session.execute(stmt)
        await self._session.commit()
        return True

    async def remove_item(self, watchlist_id: uuid.UUID, symbol: str) -> bool:
        stock_id = await self._resolve_stock_id(symbol)
        if stock_id is None:
            return False

        stmt = (
            delete(WatchlistItemModel)
            .where(WatchlistItemModel.watchlist_id == watchlist_id, WatchlistItemModel.stock_id == stock_id)
            .returning(WatchlistItemModel.id)
        )
        result = await self._session.execute(stmt)
        removed = result.first() is not None
        await self._session.commit()
        return removed

    async def _resolve_stock_id(self, symbol: str) -> uuid.UUID | None:
        stmt = select(StockModel.id).where(StockModel.symbol == symbol.strip().upper())
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_items(self, watchlist_id: uuid.UUID) -> list[WatchlistItem]:
        stmt = (
            select(StockModel.symbol, StockModel.name, WatchlistItemModel.added_at)
            .join(StockModel, StockModel.id == WatchlistItemModel.stock_id)
            .where(WatchlistItemModel.watchlist_id == watchlist_id)
            .order_by(WatchlistItemModel.added_at.asc())
        )
        result = await self._session.execute(stmt)
        return [WatchlistItem(symbol=row.symbol, name=row.name, added_at=row.added_at) for row in result]
