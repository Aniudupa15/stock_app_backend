"""Monthly Momentum Portfolio API - the validated cross-sectional edge.

- GET  /trading/momentum/ranking                       -> this month's top picks
- POST /trading/accounts/{id}/momentum/rebalance       -> rebalance the paper account into the picks
- GET  /trading/accounts/{id}/momentum/portfolio       -> current holdings + value
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from services.trading_service.api.deps import get_current_user_id, get_db_session, get_owned_account
from services.trading_service.momentum.ranking import compute_ranking
from services.trading_service.momentum.rebalance import NoDataError, _latest_closes, rebalance
from services.trading_service.persistence.models import TradingAccountModel
from services.trading_service.persistence.repositories import PositionRepository

router = APIRouter(prefix="/trading", tags=["momentum"])


@router.get("/momentum/ranking")
async def ranking(
    lookback: int = 30,
    top: int = 30,
    _user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    picks = await compute_ranking(session, lookback=lookback, top=top)
    return {"lookback": lookback, "count": len(picks), "picks": [p.as_dict() for p in picks]}


@router.post("/accounts/{account_id}/momentum/rebalance")
async def do_rebalance(
    lookback: int = 30,
    top: int = 30,
    account: TradingAccountModel = Depends(get_owned_account),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    try:
        return await rebalance(session, account, lookback=lookback, top=top)
    except NoDataError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/accounts/{account_id}/momentum/portfolio")
async def portfolio(
    account: TradingAccountModel = Depends(get_owned_account),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    positions = [p for p in await PositionRepository(session).list_for_account(account.id) if p.net_qty > 0]
    closes = await _latest_closes(session)
    holdings = []
    holdings_value = Decimal("0")
    for p in positions:
        ltp = closes.get(p.symbol, p.avg_price)
        value = Decimal(p.net_qty) * ltp
        holdings_value += value
        holdings.append(
            {
                "symbol": p.symbol,
                "qty": p.net_qty,
                "avg_price": float(p.avg_price),
                "ltp": float(ltp),
                "value": float(value),
                "pnl": float((ltp - p.avg_price) * p.net_qty),
            }
        )
    cash = Decimal(account.virtual_balance) if account.virtual_balance is not None else Decimal("0")
    return {
        "cash": float(cash),
        "holdings_value": float(holdings_value),
        "total_value": float(cash + holdings_value),
        "holdings": sorted(holdings, key=lambda h: h["value"], reverse=True),
    }
