"""Manually backfill NSE daily Bhavcopy data for a date range, without
waiting for the scheduled job. Useful for initial historical population.

    python scripts/run_price_backfill.py --from 2026-01-01 --to 2026-07-17

Iterates weekdays in the range; a date with no Bhavcopy file (a holiday) is
logged and skipped, not treated as a failure. A date that fails for a real
reason (NSE unreachable, etc.) is logged and the run continues to the next
date rather than aborting the whole backfill.
"""

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.core.exceptions import ProviderUnavailableError
from app.core.logging import configure_logging
from app.infrastructure.db.session import get_session_factory
from app.providers.nse.client import NseClient
from app.providers.nse.nse_provider import NseStockDataProvider
from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository
from app.services.price_history_service import PriceHistoryService

logger = logging.getLogger(__name__)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


async def main(from_date: date, to_date: date) -> None:
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)

    client = NseClient(settings)
    total_upserted = 0
    holidays_skipped = 0
    days_failed = 0
    try:
        provider = NseStockDataProvider(client)
        session_factory = get_session_factory()
        async with session_factory() as session:
            repository = SqlAlchemyHistoricalPriceRepository(session)
            stock_repository = SqlAlchemyStockRepository(session)
            service = PriceHistoryService(repository, provider, stock_repository)

            current = from_date
            while current <= to_date:
                if current.weekday() < 5:  # Mon-Fri only - NSE has no Bhavcopy on weekends
                    try:
                        upserted = await service.backfill_date(current)
                    except ProviderUnavailableError as exc:
                        logger.warning("%s: failed - %s", current.isoformat(), exc)
                        days_failed += 1
                    else:
                        if upserted:
                            logger.info("%s: upserted %d bars", current.isoformat(), upserted)
                            total_upserted += upserted
                        else:
                            logger.info("%s: no data (holiday)", current.isoformat())
                            holidays_skipped += 1
                current += timedelta(days=1)
    finally:
        await client.aclose()

    logger.info(
        "Backfill complete: total_upserted=%d holidays_skipped=%d days_failed=%d",
        total_upserted,
        holidays_skipped,
        days_failed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill NSE daily Bhavcopy data for a date range")
    parser.add_argument("--from", dest="from_date", type=_parse_date, required=True)
    parser.add_argument("--to", dest="to_date", type=_parse_date, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.from_date, args.to_date))
