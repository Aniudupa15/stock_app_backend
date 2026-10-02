"""Sync today's NSE Bhavcopy into historical_prices - target of the
PriceSync_1800 Windows scheduled task (companion to run_momentum_email.bat's
tasks, which depend on this data being fresh).

    python scripts/run_daily_price_sync.py

Exists because the app's in-process scheduler (run_daily_price_sync in
app/infrastructure/scheduler/jobs.py) only fires while the FastAPI server is
running - this standalone script keeps historical_prices current even when
the server isn't up, by calling the same PriceHistoryService.backfill_date
for today.
"""

import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.exceptions import ProviderUnavailableError  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.infrastructure.db.session import get_session_factory  # noqa: E402
from app.providers.nse.client import NseClient  # noqa: E402
from app.providers.nse.nse_provider import NseStockDataProvider  # noqa: E402
from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository  # noqa: E402
from app.repositories.stock_repository import SqlAlchemyStockRepository  # noqa: E402
from app.services.price_history_service import PriceHistoryService  # noqa: E402

logger = logging.getLogger(__name__)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)

    client = NseClient(settings)
    try:
        provider = NseStockDataProvider(client)
        session_factory = get_session_factory()
        async with session_factory() as session:
            repository = SqlAlchemyHistoricalPriceRepository(session)
            stock_repository = SqlAlchemyStockRepository(session)
            service = PriceHistoryService(repository, provider, stock_repository)
            today = date.today()
            try:
                upserted = await service.backfill_date(today)
            except ProviderUnavailableError as exc:
                logger.warning("%s: failed - %s", today.isoformat(), exc)
            else:
                if upserted:
                    logger.info("%s: upserted %d bars", today.isoformat(), upserted)
                else:
                    logger.info("%s: no data (holiday/weekend)", today.isoformat())
    finally:
        await client.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        logger.error("FAILED: %s: %s", type(exc).__name__, exc)
        raise
