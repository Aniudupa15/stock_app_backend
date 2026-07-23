import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import SearchHistoryEntry
from app.domain.ports import SearchHistoryRepositoryPort
from app.models.search_history import SearchHistoryModel


class SqlAlchemySearchHistoryRepository(SearchHistoryRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def log(self, user_id: uuid.UUID, query: str) -> None:
        self._session.add(SearchHistoryModel(user_id=user_id, query=query.strip()))
        await self._session.commit()

    async def list_for_user(self, user_id: uuid.UUID, limit: int, offset: int) -> list[SearchHistoryEntry]:
        stmt = (
            select(SearchHistoryModel)
            .where(SearchHistoryModel.user_id == user_id)
            .order_by(SearchHistoryModel.searched_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [SearchHistoryEntry(query=row.query, searched_at=row.searched_at) for row in result.scalars().all()]

    async def clear_for_user(self, user_id: uuid.UUID) -> int:
        stmt = delete(SearchHistoryModel).where(SearchHistoryModel.user_id == user_id)
        result = await self._session.execute(stmt)
        await self._session.commit()
        return result.rowcount or 0
