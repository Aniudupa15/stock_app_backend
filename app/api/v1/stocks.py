import uuid

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import (
    get_comparison_service,
    get_corporate_action_service,
    get_fundamentals_service,
    get_indicator_service,
    get_intraday_signal_service,
    get_long_term_signal_service,
    get_news_service,
    get_optional_user_id,
    get_price_history_service,
    get_search_history_service,
    get_stock_service,
)
from app.schemas.comparison import ComparisonOut
from app.schemas.corporate_action import CorporateActionOut
from app.schemas.fundamentals import FundamentalsOut
from app.schemas.history import HistoryOut
from app.schemas.indicators import IndicatorsOut
from app.schemas.intraday_signal import IntradaySignalOut
from app.schemas.long_term_signal import LongTermSignalOut
from app.schemas.news import NewsArticleOut
from app.schemas.stock import StockDetail, StockSearchResult
from app.services.comparison_service import ComparisonService
from app.services.corporate_action_service import CorporateActionService
from app.services.fundamentals_service import FundamentalsService
from app.services.indicator_service import IndicatorService
from app.services.intraday_signal_service import IntradaySignalService
from app.services.long_term_signal_service import LongTermSignalService
from app.services.news_service import NewsService
from app.services.price_history_service import PriceHistoryService
from app.services.search_history_service import SearchHistoryService
from app.services.stock_service import StockService

_MIN_COMPARE_SYMBOLS = 2
_MAX_COMPARE_SYMBOLS = 5

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("/search", response_model=list[StockSearchResult])
async def search_stocks(
    q: str = Query(..., min_length=1, description="Symbol or company name (partial match, with fuzzy fallback)"),
    limit: int = Query(20, ge=1, le=100),
    user_id: uuid.UUID | None = Depends(get_optional_user_id),
    service: StockService = Depends(get_stock_service),
    search_history_service: SearchHistoryService = Depends(get_search_history_service),
) -> list[StockSearchResult]:
    results = await service.search(q, limit)
    if user_id is not None:
        await search_history_service.log_best_effort(user_id, q)
    return results


@router.get("/compare", response_model=ComparisonOut)
async def compare_stocks(
    symbols: str = Query(..., description="Comma-separated symbols, 2-5 of them, e.g. RELIANCE,TCS,INFY"),
    service: ComparisonService = Depends(get_comparison_service),
) -> ComparisonOut:
    # Registered before `/{symbol}` deliberately - FastAPI matches routes in
    # declaration order, so `/stocks/compare` would otherwise be swallowed
    # by `/{symbol}` treating "compare" as a stock symbol.
    parsed = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not (_MIN_COMPARE_SYMBOLS <= len(parsed) <= _MAX_COMPARE_SYMBOLS):
        raise HTTPException(
            status_code=422,
            detail=f"Provide between {_MIN_COMPARE_SYMBOLS} and {_MAX_COMPARE_SYMBOLS} symbols, got {len(parsed)}",
        )
    return await service.compare(parsed)


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
