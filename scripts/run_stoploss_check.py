"""Daily stop-loss check for momentum paper accounts.

The monthly rebalance only ever exits a position when it drops out of the
top-N at the next rebalance - nothing checks positions in between. A single
pick can crash hard mid-month (the 30-day backtest found one that lost 63%
before its next scheduled rebalance) and ride out the full drawdown
unprotected. This runs check_stop_losses() for every account with open
momentum positions, so a loss beyond the threshold gets sold immediately
instead of waiting for the next rebalance.

    python scripts/run_stoploss_check.py

Safe to run multiple times a day (idempotent - a position already stopped
out has net_qty=0, so a repeat run finds nothing to do).
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from scripts.whatsapp_notify import notify as notify_whatsapp  # noqa: E402
from services.trading_service.momentum.rebalance import check_stop_losses  # noqa: E402
from services.trading_service.persistence.repositories import TradingAccountRepository  # noqa: E402

logger = logging.getLogger(__name__)


async def main() -> None:
    url = os.environ.get("MOMENTUM_DB_URL")
    if not url:
        from app.core.config import get_settings

        url = get_settings().DATABASE_URL
    engine = create_async_engine(url)
    sf = async_sessionmaker(bind=engine, expire_on_commit=False)
    try:
        async with sf() as s:
            account_ids = [
                r[0]
                for r in (
                    await s.execute(text("select distinct account_id from trading.positions where net_qty > 0"))
                ).all()
            ]
            total_stopped = 0
            alerts: list[str] = []
            for aid in account_ids:
                acct = await TradingAccountRepository(s).get(aid)
                if acct is None:
                    continue
                result = await check_stop_losses(s, acct)
                if result["stopped_out"]:
                    total_stopped += len(result["stopped_out"])
                    for so in result["stopped_out"]:
                        line = f"{so['symbol']} at -{so['loss_pct']:.1f}% (exit Rs{so['exit_price']:.2f})"
                        print(f"{datetime.now().isoformat()} | STOPPED OUT: {line}")
                        alerts.append(line)
            await s.commit()
            if alerts:
                notify_whatsapp("Stop-loss triggered (paper)", "\n".join(alerts))
            if total_stopped == 0:
                print(f"{datetime.now().isoformat()} | No stop-losses triggered - all positions within threshold")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"{datetime.now().isoformat()} | FAILED: {type(exc).__name__}: {exc}")
        raise
