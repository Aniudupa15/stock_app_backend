from fastapi import APIRouter, Depends, Query

from app.api.deps import (
    get_corporate_action_service,
    get_fundamentals_service,
    get_indicator_service,
    get_intraday_signal_service,
    get_long_term_signal_service,
    get_news_service,
    get_price_history_service,
    get_stock_service,
)
from app.schemas.corporate_action import CorporateActionOut
from app.schemas.fundamentals import FundamentalsOut
from app.schemas.history import HistoryOut
from app.schemas.indicators import IndicatorsOut
from app.schemas.intraday_signal import IntradaySignalOut
from app.schemas.long_term_signal import LongTermSignalOut
from app.schemas.news import NewsArticleOut
from app.schemas.stock import StockDetail, StockSearchResult
from app.services.corporate_action_service import CorporateActionService
from app.services.fundamentals_service import FundamentalsService
from app.services.indicator_service import IndicatorService
from app.services.intraday_signal_service import IntradaySignalService
from app.services.long_term_signal_service import LongTermSignalService
from app.services.news_service import NewsService
from app.services.price_history_service import PriceHistoryService
from app.services.stock_service import StockService

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("/search", response_model=list[StockSearchResult])
async def search_stocks(
    q: str = Query(..., min_length=1, description="Symbol or company name (partial match, with fuzzy fallback)"),
    limit: int = Query(20, ge=1, le=100),
    service: StockService = Depends(get_stock_service),
) -> list[StockSearchResult]:
    return await service.search(q, limit)


@router.get("/{symbol}", response_model=StockDetail)
async def get_stock_detail(
    symbol: str,
    service: StockService = Depends(get_stock_service),
) -> StockDetail:
    return await service.get_detail(symbol)


@router.get("/{symbol}/history", response_model=HistoryOut)
async def get_stock_history(
    symbol: str,
    range: str = Query(  # noqa: A002 - matches the public API param name
        "1Y", description="1D, 5D, 1M, 3M, 6M, 1Y, 3Y, 5Y, or MAX. 1D/5D are limited to daily EOD granularity."
    ),
    service: PriceHistoryService = Depends(get_price_history_service),
) -> HistoryOut:
    return await service.get_history(symbol, range)


@router.get("/{symbol}/indicators", response_model=IndicatorsOut)
async def get_stock_indicators(
    symbol: str,
    service: IndicatorService = Depends(get_indicator_service),
) -> IndicatorsOut:
    return await service.get_indicators(symbol)


@router.get("/{symbol}/corporate-actions", response_model=list[CorporateActionOut])
async def get_stock_corporate_actions(
    symbol: str,
    service: CorporateActionService = Depends(get_corporate_action_service),
) -> list[CorporateActionOut]:
    return await service.get_for_symbol(symbol)


@router.get("/{symbol}/intraday-signal", response_model=IntradaySignalOut)
async def get_stock_intraday_signal(
    symbol: str,
    service: IntradaySignalService = Depends(get_intraday_signal_service),
) -> IntradaySignalOut:
    return await service.get_signal(symbol)


@router.get("/{symbol}/fundamentals", response_model=FundamentalsOut)
async def get_stock_fundamentals(
    symbol: str,
    service: FundamentalsService = Depends(get_fundamentals_service),
) -> FundamentalsOut:
    return await service.get_fundamentals(symbol)


@router.get("/{symbol}/long-term-signal", response_model=LongTermSignalOut)
async def get_stock_long_term_signal(
    symbol: str,
    service: LongTermSignalService = Depends(get_long_term_signal_service),
) -> LongTermSignalOut:
    return await service.get_signal(symbol)


@router.get("/{symbol}/news", response_model=list[NewsArticleOut])
async def get_stock_news(
    symbol: str,
    service: NewsService = Depends(get_news_service),
) -> list[NewsArticleOut]:
    return await service.get_for_symbol(symbol)
