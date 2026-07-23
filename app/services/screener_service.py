from app.domain.entities import ScreenerFilters
from app.domain.ports import ScreenerRepositoryPort
from app.schemas.screener import ScreenerRequest, ScreenerResultOut


class ScreenerService:
    def __init__(self, repository: ScreenerRepositoryPort):
        self._repository = repository

    async def screen(self, request: ScreenerRequest) -> list[ScreenerResultOut]:
        filters = ScreenerFilters(
            rsi_below=request.rsi_below,
            rsi_above=request.rsi_above,
            price_min=request.price_min,
            price_max=request.price_max,
            above_sma_50=request.above_sma_50,
            min_volume=request.min_volume,
        )
        snapshots = await self._repository.screen(filters, request.limit)
        return [
            ScreenerResultOut(
                symbol=s.symbol,
                name=s.name,
                as_of=s.as_of,
                close=s.close,
                volume=s.volume,
                rsi_14=s.rsi_14,
                sma_50=s.sma_50,
                sma_200=s.sma_200,
            )
            for s in snapshots
        ]
