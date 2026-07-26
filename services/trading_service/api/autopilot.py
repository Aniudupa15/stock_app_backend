"""Auto-pilot API: scan the market, run the auto-pilot, get the report.

- GET  /trading/scanner                          -> today's top candidates (real data)
- POST /trading/accounts/{id}/autopilot/run      -> scan + trade the picks + report
- GET  /trading/accounts/{id}/report             -> account P&L summary
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from services.trading_service.api.deps import get_current_user_id, get_db_session, get_owned_account
from services.trading_service.autopilot.eod_report import portfolio_summary
from services.trading_service.autopilot.runner import run_autopilot
from services.trading_service.autopilot.scanner import scan_candidates
from services.trading_service.persistence.models import TradingAccountModel
from services.trading_service.persistence.repositories import TradeRepository

router = APIRouter(prefix="/trading", tags=["autopilot"])


@router.get("/scanner")
async def scanner(
    limit: int = 20,
    min_confidence: float = 60.0,
    _user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    candidates = await scan_candidates(session, min_confidence=Decimal(str(min_confidence)), limit=limit)
    return {"count": len(candidates), "candidates": [c.as_dict() for c in candidates]}


@router.post("/accounts/{account_id}/autopilot/run")
async def autopilot_run(
    top_k: int = 10,
    lookback_days: int = 180,
    account: TradingAccountModel = Depends(get_owned_account),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    return await run_autopilot(session, account, top_k=top_k, lookback_days=lookback_days)


@router.get("/accounts/{account_id}/report")
async def account_report(
    account: TradingAccountModel = Depends(get_owned_account),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    trades = await TradeRepository(session).list_for_account(account.id)
    return {"summary": portfolio_summary(trades)}
