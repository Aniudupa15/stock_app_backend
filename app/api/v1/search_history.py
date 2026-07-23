import uuid

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import get_current_user_id, get_search_history_service
from app.schemas.search_history import SearchHistoryEntryOut
from app.services.search_history_service import SearchHistoryService

router = APIRouter(prefix="/search-history", tags=["search-history"])


@router.get("", response_model=list[SearchHistoryEntryOut])
async def list_search_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: SearchHistoryService = Depends(get_search_history_service),
) -> list[SearchHistoryEntryOut]:
    return await service.list(user_id, limit, offset)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def clear_search_history(
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: SearchHistoryService = Depends(get_search_history_service),
) -> None:
    await service.clear(user_id)
