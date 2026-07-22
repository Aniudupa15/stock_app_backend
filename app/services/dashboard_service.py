from app.core.config import Settings
from app.domain.ports import CachePort
from app.schemas.dashboard import DashboardOut
from app.services.market_mover_service import MarketMoverService
from app.services.news_service import NewsService
from app.services.stock_service import StockService

_LIST_LIMIT = 10
_CACHE_KEY = "dashboard:v1"

_NOTES = [
    "'most_active' also serves as 'trending stocks' - no view-tracking exists to compute a distinct trending signal.",
    "Sector performance is not available - sector/industry data is not populated for any stock yet.",
    "'market_status' and 'indices' are empty when NSE's cookie-gated API is unreachable (best-effort, not an error).",
]


class DashboardService:
    """Composes the read-only services that already exist for each section -
    no new data access of its own, just orchestration. The composed result
    is cached (short TTL) since it's the one clearly expensive multi-query
    endpoint - a real demonstration that the Redis-backed CachePort actually
    does something, not just a wired-but-unused backend swap.
    """

    def __init__(
        self,
        stock_service: StockService,
        market_mover_service: MarketMoverService,
        news_service: NewsService,
        cache: CachePort,
        settings: Settings,
    ):
        self._stock_service = stock_service
        self._market_mover_service = market_mover_service
        self._news_service = news_service
        self._cache = cache
        self._settings = settings

    async def get_dashboard(self) -> DashboardOut:
        cached = await self._cache.get(_CACHE_KEY)
        if cached is not None:
            return cached

        dashboard = DashboardOut(
            market_status=await self._stock_service.get_market_status(),
            indices=await self._stock_service.get_indices(),
            gainers=await self._market_mover_service.get_gainers("1D", _LIST_LIMIT),
            losers=await self._market_mover_service.get_losers("1D", _LIST_LIMIT),
            most_active=await self._market_mover_service.get_most_active(_LIST_LIMIT),
            fifty_two_week_high=await self._market_mover_service.get_52_week_high(_LIST_LIMIT),
            fifty_two_week_low=await self._market_mover_service.get_52_week_low(_LIST_LIMIT),
            latest_news=await self._news_service.list_latest(None, None, limit=_LIST_LIMIT, offset=0),
            notes=_NOTES,
        )
        await self._cache.set(_CACHE_KEY, dashboard, self._settings.CACHE_DASHBOARD_TTL_SECONDS)
        return dashboard
