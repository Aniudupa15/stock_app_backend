from fastapi import APIRouter, Depends, Query

from app.api.deps import get_market_mover_service
from app.schemas.market_mover import MarketMoverOut
from app.services.market_mover_service import MarketMoverService

router = APIRouter(prefix="/market", tags=["market"])

_period_query = Query("1D", description="1D, 1W, 1M, 3M, or 1Y")
_limit_query = Query(20, ge=1, le=100)


@router.get("/gainers", response_model=list[MarketMoverOut])
async def get_gainers(
    period: str = _period_query,
    limit: int = _limit_query,
    service: MarketMoverService = Depends(get_market_mover_service),
) -> list[MarketMoverOut]:
    return await service.get_gainers(period, limit)


@router.get("/losers", response_model=list[MarketMoverOut])
async def get_losers(
    period: str = _period_query,
    limit: int = _limit_query,
    service: MarketMoverService = Depends(get_market_mover_service),
) -> list[MarketMoverOut]:
    return await service.get_losers(period, limit)


@router.get("/most-active", response_model=list[MarketMoverOut])
async def get_most_active(
    limit: int = _limit_query,
    service: MarketMoverService = Depends(get_market_mover_service),
) -> list[MarketMoverOut]:
    return await service.get_most_active(limit)


@router.get("/52-week-high", response_model=list[MarketMoverOut])
async def get_52_week_high(
    limit: int = _limit_query,
    service: MarketMoverService = Depends(get_market_mover_service),
) -> list[MarketMoverOut]:
    return await service.get_52_week_high(limit)


@router.get("/52-week-low", response_model=list[MarketMoverOut])
async def get_52_week_low(
    limit: int = _limit_query,
    service: MarketMoverService = Depends(get_market_mover_service),
) -> list[MarketMoverOut]:
    return await service.get_52_week_low(limit)
