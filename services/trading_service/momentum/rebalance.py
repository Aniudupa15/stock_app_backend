"""Monthly momentum rebalance on a paper account: sell the current holdings,
buy the new top-N equal-weight. Reuses positions/trades/equity/audit tables (no
new schema). Delivery (CNC) charges applied on every leg.
"""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from libs.charges.models import Product, Side
from libs.charges.nse_equity import compute
from libs.trading_domain.entities import Trade
from libs.trading_domain.enums import ExitReason
from services.trading_service.momentum.ranking import compute_ranking
from services.trading_service.persistence.models import TradingAccountModel
from services.trading_service.persistence.repositories import (
    AuditLogRepository,
    EquitySnapshotRepository,
    PositionRepository,
    TradeRepository,
    TradingAccountRepository,
)

_DEFAULT_CASH = Decimal("1000000")


class NoDataError(Exception):
    pass


async def _latest_closes(session: AsyncSession) -> dict[str, Decimal]:
    rows = (
        await session.execute(
            text(
                "select s.symbol, hp.close from historical_prices hp join stocks s on s.id = hp.stock_id "
                "where hp.trade_date = (select max(trade_date) from historical_prices)"
            )
        )
    ).all()
    return {symbol: Decimal(str(close)) for symbol, close in rows}


async def rebalance(session: AsyncSession, account: TradingAccountModel, *, lookback: int = 30, top: int = 10) -> dict:
    picks = await compute_ranking(session, lookback=lookback, top=top)
    if not picks:
        raise NoDataError("no momentum ranking available (insufficient price data)")
    closes = await _latest_closes(session)

    pos_repo = PositionRepository(session)
    trade_repo = TradeRepository(session)
    cash = Decimal(account.virtual_balance) if account.virtual_balance is not None else _DEFAULT_CASH

    # 1. Sell current holdings at last close.
    holdings = [p for p in await pos_repo.list_for_account(account.id) if p.net_qty > 0]
    sold = []
    for h in holdings:
        price = closes.get(h.symbol, h.avg_price)
        qty = h.net_qty
        sell_ch = compute(Side.SELL, Product.CNC, qty, price)
        buy_ch = compute(Side.BUY, Product.CNC, qty, h.avg_price)
        pnl_gross = (price - h.avg_price) * qty
        charges = sell_ch.total + buy_ch.total
        cash += qty * price - sell_ch.total
        await trade_repo.add(
            Trade(
                account_id=account.id,
                symbol=h.symbol,
                qty=qty,
                entry_price=h.avg_price,
                exit_price=price,
                pnl_gross=pnl_gross,
                charges_total=charges,
                pnl_net=pnl_gross - charges,
                exit_reason=ExitReason.SIGNAL,
            )
        )
        await pos_repo.upsert(account.id, h.symbol, "CNC", 0, Decimal("0"), h.realized_pnl + pnl_gross)
        sold.append(h.symbol)

    # 2. Buy the new top-N, equal-weight.
    budget = cash / Decimal(len(picks))
    bought = []
    for pk in picks:
        price = Decimal(str(pk.last_close))
        if price <= 0:
            continue
        qty = int(budget / (price * Decimal("1.01")))  # headroom for charges
        if qty <= 0:
            continue
        cost = qty * price + compute(Side.BUY, Product.CNC, qty, price).total
        if cost > cash:
            continue
        cash -= cost
        await pos_repo.upsert(account.id, pk.symbol, "CNC", qty, price, Decimal("0"))
        bought.append({"symbol": pk.symbol, "qty": qty, "price": float(price)})

    holdings_value = sum(
        (Decimal(str(b["qty"])) * closes.get(b["symbol"], Decimal(str(b["price"]))) for b in bought), Decimal("0")
    )
    portfolio_value = cash + holdings_value
    await TradingAccountRepository(session).update_balance(account.id, cash)
    await EquitySnapshotRepository(session).append(account.id, portfolio_value, cash)
    await AuditLogRepository(session).append(
        actor="momentum",
        event_type="REBALANCE",
        account_id=account.id,
        payload={"sold": len(sold), "bought": len(bought), "picks": [p.symbol for p in picks]},
    )
    return {
        "sold": sold,
        "bought": bought,
        "cash": str(cash.quantize(Decimal("0.01"))),
        "portfolio_value": str(portfolio_value.quantize(Decimal("0.01"))),
        "picks": [p.as_dict() for p in picks],
    }
