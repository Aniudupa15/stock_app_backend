"""Daily momentum-portfolio report -> in-app notification.

For every account holding a momentum portfolio: mark it to market at the latest
close, record a daily equity snapshot (so the curve builds), and post an in-app
notification with value, the move since the last check, and a rebalance nudge if
it's been ~a month. Runs post-market when the day's data has landed.
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.notification_repository import SqlAlchemyNotificationRepository
from services.trading_service.momentum.rebalance import _latest_closes
from services.trading_service.persistence.repositories import (
    EquitySnapshotRepository,
    PositionRepository,
    TradingAccountRepository,
)

logger = logging.getLogger(__name__)
_REBALANCE_DUE_DAYS = 28


async def generate_daily_reports(session: AsyncSession) -> int:
    """Post a momentum update notification to every account with holdings.
    Returns the number of reports sent."""
    closes = await _latest_closes(session)
    if not closes:
        return 0

    account_ids = [
        r[0]
        for r in (
            await session.execute(text("select distinct account_id from trading.positions where net_qty > 0"))
        ).all()
    ]
    notif_repo = SqlAlchemyNotificationRepository(session)
    eq_repo = EquitySnapshotRepository(session)
    sent = 0

    for account_id in account_ids:
        account = await TradingAccountRepository(session).get(account_id)
        if account is None:
            continue
        positions = [p for p in await PositionRepository(session).list_for_account(account_id) if p.net_qty > 0]
        holdings_value = sum((Decimal(p.net_qty) * closes.get(p.symbol, p.avg_price) for p in positions), Decimal("0"))
        cash = Decimal(account.virtual_balance) if account.virtual_balance is not None else Decimal("0")
        total = cash + holdings_value

        curve = await eq_repo.curve(account_id)
        prior = Decimal(curve[-1].equity) if curve else total
        change = total - prior
        start = Decimal(account.starting_balance) if account.starting_balance is not None else total
        since_start = total - start

        await eq_repo.append(account_id, total, cash)

        last_reb = (
            await session.execute(
                text("select max(ts) from trading.audit_log where account_id = :a and event_type = 'REBALANCE'"),
                {"a": account_id},
            )
        ).scalar()
        rebalance_due = last_reb is None or (date.today() - last_reb.date()).days >= _REBALANCE_DUE_DAYS

        arrow = "▲" if change >= 0 else "▼"
        overall = "+" if since_start >= 0 else "-"
        message = (
            f"Momentum portfolio: Rs{total:,.0f}  ({arrow} Rs{abs(change):,.0f} since last check). "
            f"{len(positions)} holdings, {overall}Rs{abs(since_start):,.0f} since start."
        )
        if rebalance_due:
            message += " Monthly rebalance is due - open the app and tap Rebalance."

        await notif_repo.create(account.user_id, None, "Momentum portfolio update", message)
        sent += 1

    logger.info("Momentum daily report: sent %d notification(s)", sent)
    return sent
