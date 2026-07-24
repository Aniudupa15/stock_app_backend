from fastapi import APIRouter, Depends, Query

from app.api.deps import get_dividend_service
from app.schemas.dividend import DividendRecommendationOut
from app.services.dividend_service import DividendService

router = APIRouter(prefix="/dividends", tags=["dividends"])


@router.get("", response_model=list[DividendRecommendationOut])
async def list_dividends(
    upcoming: bool = Query(False),
    sort: str = Query("ex_date", pattern="^(ex_date|yield)$"),
    limit: int = Query(50, ge=1, le=200),
    service: DividendService = Depends(get_dividend_service),
) -> list[DividendRecommendationOut]:
    return await service.list_dividends(upcoming_only=upcoming, sort=sort, limit=limit)
