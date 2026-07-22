from fastapi import APIRouter, Depends, Query

from app.api.deps import get_news_service
from app.domain.entities import NewsCategory
from app.schemas.news import NewsArticleOut
from app.services.news_service import NewsService

router = APIRouter(prefix="/news", tags=["news"])


@router.get("", response_model=list[NewsArticleOut])
async def list_news(
    category: NewsCategory | None = Query(None),
    symbol: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: NewsService = Depends(get_news_service),
) -> list[NewsArticleOut]:
    return await service.list_latest(category, symbol, limit, offset)
