"""Trading service FastAPI app. Shares the data-service's DB + JWT auth.

Deliberately no in-process trading engine yet - this exposes the account /
strategy / risk / backtest surface over the persisted schema. The live
tick-loop engine + KiteTicker wiring land in a later step behind the
multi-confirmation live-trading gate.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.exceptions import ProviderUnavailableError
from app.infrastructure.db.session import dispose_engine
from services.trading_service.api import accounts, autopilot, backtest, broker, momentum, paper, strategies


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await dispose_engine()


def create_trading_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Algo Trading Service", version="0.1.0", lifespan=_lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["health"])
    async def health() -> dict:
        return {"status": "ok", "service": "trading"}

    @app.exception_handler(ProviderUnavailableError)
    async def _provider_unavailable(_request, exc: ProviderUnavailableError):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content={"detail": "market data provider unavailable"})

    app.include_router(accounts.router)
    app.include_router(strategies.router)
    app.include_router(backtest.router)
    app.include_router(paper.router)
    app.include_router(broker.router)
    app.include_router(autopilot.router)
    app.include_router(momentum.router)

    return app
