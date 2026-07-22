from app.domain.ports import MarketMoverRepositoryPort
from app.schemas.market_mover import MarketMoverOut

# Trading-session counts, not calendar days - "1Y" is ~252 sessions, not 365.
_PERIOD_TO_SESSIONS = {
    "1D": 1,
    "1W": 5,
    "1M": 21,
    "3M": 63,
    "1Y": 252,
}
_DEFAULT_PERIOD = "1D"


def _normalize_period(period: str) -> str:
    normalized = period.upper()
    return normalized if normalized in _PERIOD_TO_SESSIONS else _DEFAULT_PERIOD


def _to_schema(movers) -> list[MarketMoverOut]:
    return [
        MarketMoverOut(
            symbol=m.symbol,
            name=m.name,
            last_price=m.last_price,
            change=m.change,
            change_percent=m.change_percent,
            volume=m.volume,
        )
        for m in movers
    ]


class MarketMoverService:
    def __init__(self, repository: MarketMoverRepositoryPort):
        self._repository = repository

    async def get_gainers(self, period: str, limit: int) -> list[MarketMoverOut]:
        sessions = _PERIOD_TO_SESSIONS[_normalize_period(period)]
        movers = await self._repository.get_top_movers("gainers", sessions, limit)
        return _to_schema(movers)

    async def get_losers(self, period: str, limit: int) -> list[MarketMoverOut]:
        sessions = _PERIOD_TO_SESSIONS[_normalize_period(period)]
        movers = await self._repository.get_top_movers("losers", sessions, limit)
        return _to_schema(movers)

    async def get_most_active(self, limit: int) -> list[MarketMoverOut]:
        movers = await self._repository.get_most_active(limit)
        return _to_schema(movers)

    async def get_52_week_high(self, limit: int) -> list[MarketMoverOut]:
        movers = await self._repository.get_52_week_extremes("high", limit)
        return _to_schema(movers)

    async def get_52_week_low(self, limit: int) -> list[MarketMoverOut]:
        movers = await self._repository.get_52_week_extremes("low", limit)
        return _to_schema(movers)
