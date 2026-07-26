from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.v1.router import router as v1_router
from app.cache.memory_cache import InMemoryCache
from app.cache.redis_cache import RedisCache
from app.core.config import get_settings
from app.core.exceptions import (
    AlertNotFoundError,
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
    InvalidRefreshTokenError,
    NotificationNotFoundError,
    PortfolioNotFoundError,
    ProviderUnavailableError,
    StockNotFoundError,
    UserNotFoundError,
    WatchlistNotFoundError,
)
from app.core.logging import configure_logging
from app.infrastructure.db.session import dispose_engine
from app.infrastructure.live.broadcaster import Broadcaster
from app.infrastructure.live.connection_manager import ConnectionManager
from app.infrastructure.scheduler.bootstrap import start_scheduler, stop_scheduler
from app.providers.nse.client import NseClient


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)

    app.state.settings = settings
    app.state.cache = RedisCache(settings.REDIS_URL) if settings.CACHE_BACKEND == "redis" else InMemoryCache()
    app.state.nse_client = NseClient(settings)
    app.state.scheduler = start_scheduler(settings) if settings.SCHEDULER_ENABLED else None

    app.state.live_connection_manager = ConnectionManager()
    app.state.live_broadcaster = Broadcaster(
        settings, app.state.cache, app.state.nse_client, app.state.live_connection_manager
    )
    app.state.live_broadcaster.start()

    yield

    await app.state.live_broadcaster.stop()
    if app.state.scheduler is not None:
        stop_scheduler(app.state.scheduler)
    if isinstance(app.state.cache, RedisCache):
        await app.state.cache.aclose()
    await app.state.nse_client.aclose()
    await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.APP_NAME, version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(StockNotFoundError)
    async def _stock_not_found_handler(request: Request, exc: StockNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ProviderUnavailableError)
    async def _provider_unavailable_handler(request: Request, exc: ProviderUnavailableError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(WatchlistNotFoundError)
    async def _watchlist_not_found_handler(request: Request, exc: WatchlistNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(PortfolioNotFoundError)
    async def _portfolio_not_found_handler(request: Request, exc: PortfolioNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(AlertNotFoundError)
    async def _alert_not_found_handler(request: Request, exc: AlertNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(NotificationNotFoundError)
    async def _notification_not_found_handler(request: Request, exc: NotificationNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(EmailAlreadyRegisteredError)
    async def _email_already_registered_handler(request: Request, exc: EmailAlreadyRegisteredError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(InvalidCredentialsError)
    async def _invalid_credentials_handler(request: Request, exc: InvalidCredentialsError) -> JSONResponse:
        return JSONResponse(status_code=401, content={"detail": str(exc)}, headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(InvalidRefreshTokenError)
    async def _invalid_refresh_token_handler(request: Request, exc: InvalidRefreshTokenError) -> JSONResponse:
        return JSONResponse(status_code=401, content={"detail": str(exc)}, headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(UserNotFoundError)
    async def _user_not_found_handler(request: Request, exc: UserNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    app.include_router(v1_router, prefix="/api/v1")

    # Mount the trading-platform API (/trading/*) into the same app so a single
    # deployment serves both the data API and the trading API - they already
    # share this database and JWT auth. For LIVE trading, split trading_service
    # into its own deployment with a static egress IP (see docs/trading-platform).
    from services.trading_service.api import accounts as _trading_accounts
    from services.trading_service.api import autopilot as _trading_autopilot
    from services.trading_service.api import backtest as _trading_backtest
    from services.trading_service.api import broker as _trading_broker
    from services.trading_service.api import paper as _trading_paper
    from services.trading_service.api import strategies as _trading_strategies

    app.include_router(_trading_accounts.router)
    app.include_router(_trading_strategies.router)
    app.include_router(_trading_backtest.router)
    app.include_router(_trading_paper.router)
    app.include_router(_trading_broker.router)
    app.include_router(_trading_autopilot.router)

    # Request count/latency/in-flight gauges at GET /metrics, standard
    # Prometheus exposition format - unauthenticated by convention (scraped
    # by infrastructure, not end users).
    Instrumentator().instrument(app).expose(app, include_in_schema=False)

    return app


app = create_app()
