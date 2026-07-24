import asyncio
import logging

from app.core.config import Settings
from app.domain.ports import CachePort
from app.infrastructure.db.session import get_session_factory
from app.infrastructure.live.connection_manager import ConnectionManager
from app.providers.nse.client import NseClient
from app.providers.nse.nse_provider import NseStockDataProvider
from app.repositories.market_mover_repository import SqlAlchemyMarketMoverRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository
from app.services.market_mover_service import MarketMoverService
from app.services.stock_service import StockService

logger = logging.getLogger(__name__)

_MARKET_SNAPSHOT_LIMIT = 10
_MARKET_SNAPSHOT_PERIOD = "1D"


class Broadcaster:
    """Two independent background loops, started in the app lifespan
    alongside the existing APScheduler: one pushes live per-symbol quotes to
    whichever connections are subscribed to that symbol (reusing
    `StockService.get_detail` - the exact same cache/rate-limit/error-handling
    path the REST stock-detail endpoint uses, not a duplicated fetch), the
    other pushes a market-movers snapshot to any connection subscribed to
    `MARKET_CHANNEL`. Deliberately does no per-portfolio/per-watchlist
    aggregation - clients already hold their own items/holdings and patch
    them locally from the `quote` events they receive.
    """

    def __init__(
        self, settings: Settings, cache: CachePort, nse_client: NseClient, connection_manager: ConnectionManager
    ):
        self._settings = settings
        self._cache = cache
        self._nse_client = nse_client
        self._connection_manager = connection_manager
        self._quote_task: asyncio.Task | None = None
        self._market_task: asyncio.Task | None = None

    def start(self) -> None:
        self._quote_task = asyncio.create_task(self._quote_loop())
        self._market_task = asyncio.create_task(self._market_loop())

    async def stop(self) -> None:
        for task in (self._quote_task, self._market_task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _quote_loop(self) -> None:
        while True:
            try:
                await self._broadcast_quotes()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Live quote broadcast tick failed")
            await asyncio.sleep(self._settings.LIVE_QUOTE_INTERVAL_SECONDS)

    async def _market_loop(self) -> None:
        while True:
            try:
                await self._broadcast_market_snapshot()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Live market snapshot broadcast tick failed")
            await asyncio.sleep(self._settings.LIVE_MARKET_INTERVAL_SECONDS)

    async def _broadcast_quotes(self) -> None:
        symbols = self._connection_manager.all_subscribed_symbols()
        if not symbols:
            return

        provider = NseStockDataProvider(self._nse_client)
        session_factory = get_session_factory()
        async with session_factory() as session:
            stock_repository = SqlAlchemyStockRepository(session)
            stock_service = StockService(stock_repository, provider, self._cache, self._settings)

            for symbol in symbols:
                connections = self._connection_manager.connections_for_symbol(symbol)
                if not connections:
                    continue
                try:
                    detail = await stock_service.get_detail(symbol)
                except Exception:
                    logger.warning("Live quote broadcast: skipping %s (unexpected error)", symbol, exc_info=True)
                    continue

                if detail.quote is None:
                    message = {
                        "type": "quote_unavailable",
                        "symbol": symbol,
                        "reason": detail.quote_unavailable_reason,
                    }
                else:
                    message = {
                        "type": "quote",
                        "symbol": symbol,
                        "data": detail.quote.model_dump(mode="json"),
                    }

                for ws in connections:
                    await self._connection_manager.send_json(ws, message)

    async def _broadcast_market_snapshot(self) -> None:
        if not self._connection_manager.is_market_subscribed():
            return

        session_factory = get_session_factory()
        async with session_factory() as session:
            repository = SqlAlchemyMarketMoverRepository(session)
            service = MarketMoverService(repository)
            gainers = await service.get_gainers(_MARKET_SNAPSHOT_PERIOD, _MARKET_SNAPSHOT_LIMIT)
            losers = await service.get_losers(_MARKET_SNAPSHOT_PERIOD, _MARKET_SNAPSHOT_LIMIT)
            most_active = await service.get_most_active(_MARKET_SNAPSHOT_LIMIT)

        message = {
            "type": "market_snapshot",
            "data": {
                "gainers": [m.model_dump(mode="json") for m in gainers],
                "losers": [m.model_dump(mode="json") for m in losers],
                "most_active": [m.model_dump(mode="json") for m in most_active],
            },
        }
        for ws in self._connection_manager.market_connections():
            await self._connection_manager.send_json(ws, message)
