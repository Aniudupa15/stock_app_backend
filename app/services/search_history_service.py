import logging
import uuid

from app.domain.ports import SearchHistoryRepositoryPort
from app.schemas.search_history import SearchHistoryEntryOut

logger = logging.getLogger(__name__)


class SearchHistoryService:
    def __init__(self, repository: SearchHistoryRepositoryPort):
        self._repository = repository

    async def log_best_effort(self, user_id: uuid.UUID, query: str) -> None:
        """Never raises - a logging failure must not turn a successful
        search into an error response for the caller.
        """
        try:
            await self._repository.log(user_id, query)
        except Exception:
            logger.warning("failed to log search history for user %s", user_id, exc_info=True)

    async def list(self, user_id: uuid.UUID, limit: int, offset: int) -> list[SearchHistoryEntryOut]:
        entries = await self._repository.list_for_user(user_id, limit, offset)
        return [SearchHistoryEntryOut(query=e.query, searched_at=e.searched_at) for e in entries]

    async def clear(self, user_id: uuid.UUID) -> int:
        return await self._repository.clear_for_user(user_id)
