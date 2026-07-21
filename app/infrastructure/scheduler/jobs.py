import logging
from datetime import date, timedelta

from sqlalchemy import func, select

from app.core.config import Settings
from app.infrastructure.db.session import get_session_factory
from app.models.financial_result import FinancialResultModel
from app.models.stock import StockModel
from app.providers.nse.client import NseClient
from app.providers.nse.nse_provider import NseStockDataProvider
from app.repositories.corporate_action_repository import SqlAlchemyCorporateActionRepository
from app.repositories.financial_result_repository import SqlAlchemyFinancialResultRepository
from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository
from app.services.corporate_action_service import CorporateActionService
from app.services.financial_results_sync_service import FinancialResultsSyncService
from app.services.price_history_service import PriceHistoryService
from app.services.universe_sync_service import UniverseSyncService

logger = logging.getLogger(__name__)

_CORPORATE_ACTIONS_LOOKBACK_DAYS = 30
_CORPORATE_ACTIONS_LOOKAHEAD_DAYS = 90
_FINANCIAL_RESULTS_STALE_DAYS = 80
_FINANCIAL_RESULTS_MAX_SYMBOLS_PER_RUN = 200


async def run_universe_sync(settings: Settings) -> None:
    """Scheduled job body: replaces any hardcoded stock list with a fresh NSE pull.

    Owns its own NseClient (short-lived, one per run) rather than reusing the
    app-wide client, so a stuck sync can't starve the request-serving client's
    rate limiter/circuit breaker state.
    """
    client = NseClient(settings)
    try:
        provider = NseStockDataProvider(client)
        session_factory = get_session_factory()
        async with session_factory() as session:
            repository = SqlAlchemyStockRepository(session)
            service = UniverseSyncService(provider, repository)
            try:
                result = await service.sync_equity_universe()
                logger.info("Scheduled universe sync succeeded: %s", result)
            except Exception:
                logger.exception("Scheduled universe sync failed")
    finally:
        await client.aclose()


async def run_daily_price_sync(settings: Settings) -> None:
    """Scheduled job body: backfills today's Bhavcopy into `historical_prices`.

    If today isn't a trading day (weekend/holiday), the NSE archive simply has
    no file for it - `PriceHistoryService.backfill_date` treats that as a
    normal 0-rows result, not an error.
    """
    client = NseClient(settings)
    try:
        provider = NseStockDataProvider(client)
        session_factory = get_session_factory()
        async with session_factory() as session:
            repository = SqlAlchemyHistoricalPriceRepository(session)
            stock_repository = SqlAlchemyStockRepository(session)
            service = PriceHistoryService(repository, provider, stock_repository)
            try:
                upserted = await service.backfill_date(date.today())
                logger.info("Scheduled daily price sync succeeded: upserted=%d", upserted)
            except Exception:
                logger.exception("Scheduled daily price sync failed")
    finally:
        await client.aclose()


async def run_corporate_actions_sync(settings: Settings) -> None:
    """Scheduled job body: syncs a rolling window of corporate actions.

    Best-effort - this hits NSE's cookie-gated API (less reliable than the
    static archives used elsewhere), so failures here just mean the API keeps
    serving whatever was last synced successfully, not an outage.
    """
    client = NseClient(settings)
    try:
        provider = NseStockDataProvider(client)
        session_factory = get_session_factory()
        async with session_factory() as session:
            repository = SqlAlchemyCorporateActionRepository(session)
            stock_repository = SqlAlchemyStockRepository(session)
            service = CorporateActionService(repository, provider, stock_repository)
            try:
                from_date = date.today() - timedelta(days=_CORPORATE_ACTIONS_LOOKBACK_DAYS)
                to_date = date.today() + timedelta(days=_CORPORATE_ACTIONS_LOOKAHEAD_DAYS)
                upserted = await service.sync(from_date, to_date)
                logger.info("Scheduled corporate actions sync succeeded: upserted=%d", upserted)
            except Exception:
                logger.exception("Scheduled corporate actions sync failed")
    finally:
        await client.aclose()


async def run_financial_results_sync(settings: Settings) -> None:
    """Scheduled job body: refreshes quarterly financial results for active
    stocks whose stored data is missing or stale (> 80 days old).

    Financial results only change ~4x/year per company, and there are
    thousands of active stocks - an unconditional full pass every run would
    be extremely wasteful given NSE's per-request rate limiting. Also caps
    itself to `_FINANCIAL_RESULTS_MAX_SYMBOLS_PER_RUN` symbols so a single run
    has bounded runtime; a large initial backlog clears over several
    scheduled runs rather than one multi-hour job. Per-symbol failures are
    logged and skipped, not fatal to the run.
    """
    client = NseClient(settings)
    try:
        provider = NseStockDataProvider(client)
        session_factory = get_session_factory()
        async with session_factory() as session:
            financial_repository = SqlAlchemyFinancialResultRepository(session)
            sync_service = FinancialResultsSyncService(provider, financial_repository)

            stale_cutoff = date.today() - timedelta(days=_FINANCIAL_RESULTS_STALE_DAYS)
            result = await session.execute(
                select(StockModel.symbol)
                .outerjoin(FinancialResultModel, FinancialResultModel.stock_id == StockModel.id)
                .where(StockModel.is_active.is_(True))
                .group_by(StockModel.symbol)
                .having(
                    (func.max(FinancialResultModel.period_end).is_(None))
                    | (func.max(FinancialResultModel.period_end) < stale_cutoff)
                )
                .limit(_FINANCIAL_RESULTS_MAX_SYMBOLS_PER_RUN)
            )
            symbols = [row[0] for row in result]

            logger.info("Financial results sync: %d symbols need refresh this run", len(symbols))
            total_upserted = 0
            failed = 0
            for symbol in symbols:
                try:
                    total_upserted += await sync_service.sync_symbol(symbol)
                except Exception:
                    failed += 1
                    logger.exception("Financial results sync failed for %s", symbol)

            logger.info(
                "Financial results sync complete: symbols=%d upserted=%d failed=%d",
                len(symbols),
                total_upserted,
                failed,
            )
    finally:
        await client.aclose()
