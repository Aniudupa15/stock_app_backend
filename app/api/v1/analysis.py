from decimal import Decimal

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_analysis_service
from app.schemas.analysis import IntradayRecommendationOut, LongTermRecommendationOut
from app.services.analysis_service import AnalysisService

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("/intraday", response_model=list[IntradayRecommendationOut])
async def get_intraday_recommendations(
    limit: int = Query(50, ge=1, le=200),
    min_confidence: Decimal = Query(Decimal("0")),
    service: AnalysisService = Depends(get_analysis_service),
) -> list[IntradayRecommendationOut]:
    return await service.get_top_intraday(min_confidence, limit)


@router.get("/long-term", response_model=list[LongTermRecommendationOut])
async def get_long_term_recommendations(
    limit: int = Query(50, ge=1, le=200),
    min_confidence: int = Query(0, ge=0, le=100),
    tenure: str | None = Query(None, pattern="^(6 Months|1 Year|3 Years|5 Years)$"),
    service: AnalysisService = Depends(get_analysis_service),
) -> list[LongTermRecommendationOut]:
    return await service.get_top_long_term(min_confidence, tenure, limit)
