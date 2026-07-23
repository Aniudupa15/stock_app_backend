from decimal import Decimal

from app.domain.ports import MarketMoverRepositoryPort
from app.schemas.heatmap import HeatmapOut, HeatmapTileOut
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

_STRONG_MOVE_THRESHOLD = Decimal("3")

_HEATMAP_NOTE = (
    "Tile size is based on trading volume (a proxy for market activity) - "
    "true market capitalization is not available for any stock yet, so "
    "this reflects activity, not company size."
)


def _bucket_for(change_percent: Decimal | None) -> str:
    if change_percent is None:
        return "UNKNOWN"
    if change_percent >= _STRONG_MOVE_THRESHOLD:
        return "STRONG_GAIN"
    if change_percent > 0:
        return "GAIN"
    if change_percent == 0:
        return "FLAT"
    if change_percent > -_STRONG_MOVE_THRESHOLD:
        return "LOSS"
    return "STRONG_LOSS"


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

    async def get_heatmap(self, limit: int) -> HeatmapOut:
        """Reuses the exact same query as most-active (latest volume,
        descending) - a heatmap is that same data, presented as sized/
        colored tiles instead of a ranked list, so no new repository query
        is needed.
        """
        movers = await self._repository.get_most_active(limit)
        tiles = [
            HeatmapTileOut(
                symbol=m.symbol,
                name=m.name,
                last_price=m.last_price,
                change_percent=m.change_percent,
                volume=m.volume,
                bucket=_bucket_for(m.change_percent),
            )
            for m in movers
        ]
        return HeatmapOut(tiles=tiles, notes=[_HEATMAP_NOTE])
