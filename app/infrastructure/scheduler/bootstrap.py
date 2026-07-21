import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core.config import Settings
from app.infrastructure.scheduler.jobs import (
    run_corporate_actions_sync,
    run_daily_price_sync,
    run_financial_results_sync,
    run_universe_sync,
)

logger = logging.getLogger(__name__)

_IST_TIMEZONE = "Asia/Kolkata"


def start_scheduler(settings: Settings) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=_IST_TIMEZONE)
    scheduler.add_job(
        run_universe_sync,
        trigger=CronTrigger(hour=settings.UNIVERSE_SYNC_HOUR_IST, minute=settings.UNIVERSE_SYNC_MINUTE_IST),
        args=[settings],
        id="universe_sync",
        replace_existing=True,
    )
    scheduler.add_job(
        run_daily_price_sync,
        trigger=CronTrigger(hour=settings.PRICE_SYNC_HOUR_IST, minute=settings.PRICE_SYNC_MINUTE_IST),
        args=[settings],
        id="daily_price_sync",
        replace_existing=True,
    )
    scheduler.add_job(
        run_corporate_actions_sync,
        trigger=CronTrigger(
            hour=settings.CORPORATE_ACTIONS_SYNC_HOUR_IST, minute=settings.CORPORATE_ACTIONS_SYNC_MINUTE_IST
        ),
        args=[settings],
        id="corporate_actions_sync",
        replace_existing=True,
    )
    scheduler.add_job(
        run_financial_results_sync,
        trigger=CronTrigger(
            hour=settings.FINANCIAL_RESULTS_SYNC_HOUR_IST, minute=settings.FINANCIAL_RESULTS_SYNC_MINUTE_IST
        ),
        args=[settings],
        id="financial_results_sync",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Scheduler started: universe_sync %02d:%02d IST, daily_price_sync %02d:%02d IST, "
        "corporate_actions_sync %02d:%02d IST, financial_results_sync %02d:%02d IST",
        settings.UNIVERSE_SYNC_HOUR_IST,
        settings.UNIVERSE_SYNC_MINUTE_IST,
        settings.PRICE_SYNC_HOUR_IST,
        settings.PRICE_SYNC_MINUTE_IST,
        settings.CORPORATE_ACTIONS_SYNC_HOUR_IST,
        settings.CORPORATE_ACTIONS_SYNC_MINUTE_IST,
        settings.FINANCIAL_RESULTS_SYNC_HOUR_IST,
        settings.FINANCIAL_RESULTS_SYNC_MINUTE_IST,
    )
    return scheduler


def stop_scheduler(scheduler: AsyncIOScheduler) -> None:
    scheduler.shutdown(wait=False)
