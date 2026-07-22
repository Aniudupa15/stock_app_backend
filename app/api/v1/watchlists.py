import uuid

from fastapi import APIRouter, Depends, status

from app.api.deps import get_current_user_id, get_watchlist_service
from app.schemas.watchlist import WatchlistAddSymbol, WatchlistCreate, WatchlistDetailOut, WatchlistOut
from app.services.watchlist_service import WatchlistService

router = APIRouter(prefix="/watchlists", tags=["watchlists"])


@router.post("", response_model=WatchlistOut, status_code=status.HTTP_201_CREATED)
async def create_watchlist(
    body: WatchlistCreate,
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: WatchlistService = Depends(get_watchlist_service),
) -> WatchlistOut:
    return await service.create(user_id, body.name)


@router.get("", response_model=list[WatchlistOut])
async def list_watchlists(
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: WatchlistService = Depends(get_watchlist_service),
) -> list[WatchlistOut]:
    return await service.list(user_id)


@router.get("/{watchlist_id}", response_model=WatchlistDetailOut)
async def get_watchlist(
    watchlist_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: WatchlistService = Depends(get_watchlist_service),
) -> WatchlistDetailOut:
    return await service.get_detail(user_id, watchlist_id)


@router.delete("/{watchlist_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_watchlist(
    watchlist_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: WatchlistService = Depends(get_watchlist_service),
) -> None:
    await service.delete(user_id, watchlist_id)


@router.post("/{watchlist_id}/items", response_model=WatchlistDetailOut)
async def add_watchlist_item(
    watchlist_id: uuid.UUID,
    body: WatchlistAddSymbol,
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: WatchlistService = Depends(get_watchlist_service),
) -> WatchlistDetailOut:
    return await service.add_symbol(user_id, watchlist_id, body.symbol)


@router.delete("/{watchlist_id}/items/{symbol}", response_model=WatchlistDetailOut)
async def remove_watchlist_item(
    watchlist_id: uuid.UUID,
    symbol: str,
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: WatchlistService = Depends(get_watchlist_service),
) -> WatchlistDetailOut:
    return await service.remove_symbol(user_id, watchlist_id, symbol)
