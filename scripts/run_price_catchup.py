"""Catch up any missing recent trading days in historical_prices.

Target of the DailyMoversFollowup_0915 task: a safety net for when the 18:00
PriceSync_1800 job fails overnight (laptop asleep/offline, etc.) - by 9:15
the network is normally back, so this fills the gap before the 9:15 movers
report re-sends with (hopefully) fresher data.

    python scripts/run_price_catchup.py

Backfills from (latest trade_date in DB + 1) through yesterday, inclusive,
weekdays only. If already caught up through yesterday, does nothing.
"""

import asyncio
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

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

            latest = (await session.execute(text("select max(trade_date) from historical_prices"))).scalar()
            yesterday = date.today() - timedelta(days=1)
            current = (latest + timedelta(days=1)) if latest else yesterday
            if current > yesterday:
                logger.info("already caught up through %s - nothing to do", latest)
                return

            total_upserted = 0
            while current <= yesterday:
                if current.weekday() < 5:  # Mon-Fri only
                    try:
                        upserted = await service.backfill_date(current)
                    except ProviderUnavailableError as exc:
                        logger.warning("%s: failed - %s", current.isoformat(), exc)
                    else:
                        if upserted:
                            logger.info("%s: upserted %d bars", current.isoformat(), upserted)
                            total_upserted += upserted
                        else:
                            logger.info("%s: no data (holiday)", current.isoformat())
                current += timedelta(days=1)
            logger.info("Catch-up complete: total_upserted=%d", total_upserted)
    finally:
        await client.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        logger.error("FAILED: %s: %s", type(exc).__name__, exc)
        raise
