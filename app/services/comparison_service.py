from app.schemas.comparison import ComparisonEntryOut, ComparisonOut
from app.services.fundamentals_service import FundamentalsService
from app.services.indicator_service import IndicatorService
from app.services.stock_service import StockService


class ComparisonService:
    """Pure composition of three existing services, one call per symbol - no
    new data access, no new business logic. `StockService.get_detail`
    raising `StockNotFoundError` for any unknown symbol is exactly the
    behavior we want here too (propagates as-is, naming the bad symbol).
    """

    def __init__(
        self,
        stock_service: StockService,
        indicator_service: IndicatorService,
        fundamentals_service: FundamentalsService,
    ):
        self._stock_service = stock_service
        self._indicator_service = indicator_service
        self._fundamentals_service = fundamentals_service

    async def compare(self, symbols: list[str]) -> ComparisonOut:
        entries = []
        for symbol in symbols:
            detail = await self._stock_service.get_detail(symbol)
            indicators = await self._indicator_service.get_indicators(symbol)
            fundamentals = await self._fundamentals_service.get_fundamentals(symbol)
            entries.append(ComparisonEntryOut(detail=detail, indicators=indicators, fundamentals=fundamentals))
        return ComparisonOut(entries=entries)
