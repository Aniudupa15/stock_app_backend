"""Persisted paper-trading endpoint: run a saved strategy over history and
commit the trade journal + equity curve + balance to the account."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from services.trading_service.api.deps import get_current_user_id, get_db_session, get_owned_account
from services.trading_service.api.schemas import PaperRunOut, PaperRunRequest
from services.trading_service.paper_session import InsufficientHistoryError, metrics_to_dict, run_paper_session
from services.trading_service.persistence.models import TradingAccountModel
from services.trading_service.persistence.repositories import StrategyRepository
from services.trading_service.strategy_mapping import strategy_from_model

router = APIRouter(prefix="/trading", tags=["paper"])


@router.post("/accounts/{account_id}/paper-run", response_model=PaperRunOut)
async def paper_run(
    body: PaperRunRequest,
    account: TradingAccountModel = Depends(get_owned_account),
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
) -> PaperRunOut:
    model = await StrategyRepository(session).get(body.strategy_id, user_id)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="strategy not found")

    strategy = strategy_from_model(model)
    try:
        result = await run_paper_session(session, account, strategy, body.symbol, body.from_date, body.to_date)
    except InsufficientHistoryError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return PaperRunOut(
        account_id=account.id,
        symbol=body.symbol.strip().upper(),
        bars=len(result.equity_curve),
        trades=len(result.trades),
        net_pnl=result.metrics.net_pnl,
        final_equity=result.final_equity,
        metrics=metrics_to_dict(result.metrics),
    )
