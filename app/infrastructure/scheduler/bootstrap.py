import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.core.config import Settings
from app.infrastructure.scheduler.jobs import (
    run_alert_evaluation,
    run_corporate_actions_sync,
    run_daily_price_sync,
    run_financial_results_sync,
    run_indicator_snapshot_sync,
    run_ipo_sync,
    run_momentum_report,
    run_news_sync,
    run_signal_snapshot_sync,
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
    scheduler.add_job(
        run_news_sync,
        trigger=IntervalTrigger(minutes=settings.NEWS_SYNC_INTERVAL_MINUTES),
        args=[settings],
        id="news_sync",
        replace_existing=True,
    )
    scheduler.add_job(
        run_alert_evaluation,
        trigger=IntervalTrigger(minutes=settings.ALERT_EVALUATION_INTERVAL_MINUTES),
        args=[settings],
        id="alert_evaluation",
        replace_existing=True,
    )
    scheduler.add_job(
        run_indicator_snapshot_sync,
        trigger=CronTrigger(
            hour=settings.INDICATOR_SNAPSHOT_SYNC_HOUR_IST, minute=settings.INDICATOR_SNAPSHOT_SYNC_MINUTE_IST
        ),
        args=[settings],
        id="indicator_snapshot_sync",
        replace_existing=True,
    )
    scheduler.add_job(
        run_ipo_sync,
        trigger=CronTrigger(hour=settings.IPO_SYNC_HOUR_IST, minute=settings.IPO_SYNC_MINUTE_IST),
        args=[settings],
        id="ipo_sync",
        replace_existing=True,
    )
    scheduler.add_job(
        run_signal_snapshot_sync,
        trigger=CronTrigger(
            hour=settings.SIGNAL_SNAPSHOT_SYNC_HOUR_IST, minute=settings.SIGNAL_SNAPSHOT_SYNC_MINUTE_IST
        ),
        args=[settings],
        id="signal_snapshot_sync",
        replace_existing=True,
    )
    scheduler.add_job(
        run_momentum_report,
        trigger=CronTrigger(hour=settings.MOMENTUM_REPORT_HOUR_IST, minute=settings.MOMENTUM_REPORT_MINUTE_IST),
        args=[settings],
        id="momentum_report",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Scheduler started: universe_sync %02d:%02d IST, daily_price_sync %02d:%02d IST, "
        "corporate_actions_sync %02d:%02d IST, financial_results_sync %02d:%02d IST, "
        "news_sync every %d min, alert_evaluation every %d min, indicator_snapshot_sync %02d:%02d IST, "
        "ipo_sync %02d:%02d IST, signal_snapshot_sync %02d:%02d IST",
        settings.UNIVERSE_SYNC_HOUR_IST,
        settings.UNIVERSE_SYNC_MINUTE_IST,
        settings.PRICE_SYNC_HOUR_IST,
        settings.PRICE_SYNC_MINUTE_IST,
        settings.CORPORATE_ACTIONS_SYNC_HOUR_IST,
        settings.CORPORATE_ACTIONS_SYNC_MINUTE_IST,
        settings.FINANCIAL_RESULTS_SYNC_HOUR_IST,
        settings.FINANCIAL_RESULTS_SYNC_MINUTE_IST,
        settings.NEWS_SYNC_INTERVAL_MINUTES,
        settings.ALERT_EVALUATION_INTERVAL_MINUTES,
        settings.INDICATOR_SNAPSHOT_SYNC_HOUR_IST,
        settings.INDICATOR_SNAPSHOT_SYNC_MINUTE_IST,
        settings.IPO_SYNC_HOUR_IST,
        settings.IPO_SYNC_MINUTE_IST,
        settings.SIGNAL_SNAPSHOT_SYNC_HOUR_IST,
        settings.SIGNAL_SNAPSHOT_SYNC_MINUTE_IST,
    )
    return scheduler


def stop_scheduler(scheduler: AsyncIOScheduler) -> None:
    scheduler.shutdown(wait=False)
