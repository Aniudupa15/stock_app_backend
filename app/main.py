from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import router as v1_router
from app.cache.memory_cache import InMemoryCache
from app.core.config import get_settings
from app.core.exceptions import ProviderUnavailableError, StockNotFoundError
from app.core.logging import configure_logging
from app.infrastructure.db.session import dispose_engine
from app.infrastructure.scheduler.bootstrap import start_scheduler, stop_scheduler
from app.providers.nse.client import NseClient


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)

    app.state.settings = settings
    app.state.cache = InMemoryCache()
    app.state.nse_client = NseClient(settings)
    app.state.scheduler = start_scheduler(settings) if settings.SCHEDULER_ENABLED else None

    yield

    if app.state.scheduler is not None:
        stop_scheduler(app.state.scheduler)
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

    app.include_router(v1_router, prefix="/api/v1")

    return app


app = create_app()
