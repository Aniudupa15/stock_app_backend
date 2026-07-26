"""Account, risk-profile, kill-switch, and portfolio-read endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from services.trading_service.api.deps import get_current_user_id, get_db_session, get_owned_account
from services.trading_service.api.schemas import (
    AccountCreate,
    AccountOut,
    EquityPointOut,
    KillSwitchIn,
    OrderOut,
    PositionOut,
    RiskProfileIn,
    RiskProfileOut,
    TradeOut,
)
from services.trading_service.persistence.models import TradingAccountModel
from services.trading_service.persistence.repositories import (
    EquitySnapshotRepository,
    OrderRepository,
    PositionRepository,
    RiskProfileRepository,
    TradeRepository,
    TradingAccountRepository,
)

router = APIRouter(prefix="/trading", tags=["trading"])


@router.post("/accounts", response_model=AccountOut, status_code=status.HTTP_201_CREATED)
async def create_account(
    body: AccountCreate,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
) -> TradingAccountModel:
    return await TradingAccountRepository(session).create(user_id, body.mode, body.starting_balance)


@router.get("/accounts", response_model=list[AccountOut])
async def list_accounts(
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
) -> list[TradingAccountModel]:
    return await TradingAccountRepository(session).list_for_user(user_id)


@router.get("/accounts/{account_id}", response_model=AccountOut)
async def get_account(account: TradingAccountModel = Depends(get_owned_account)) -> TradingAccountModel:
    return account


@router.put("/accounts/{account_id}/risk", response_model=RiskProfileOut)
async def upsert_risk(
    body: RiskProfileIn,
    account: TradingAccountModel = Depends(get_owned_account),
    session: AsyncSession = Depends(get_db_session),
):
    fields = body.model_dump(exclude_none=True)
    return await RiskProfileRepository(session).upsert(account.id, **fields)


@router.get("/accounts/{account_id}/risk", response_model=RiskProfileOut)
async def get_risk(
    account: TradingAccountModel = Depends(get_owned_account),
    session: AsyncSession = Depends(get_db_session),
):
    profile = await RiskProfileRepository(session).get(account.id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no risk profile configured")
    return profile


@router.post("/accounts/{account_id}/kill-switch", response_model=RiskProfileOut)
async def set_kill_switch(
    body: KillSwitchIn,
    account: TradingAccountModel = Depends(get_owned_account),
    session: AsyncSession = Depends(get_db_session),
):
    repo = RiskProfileRepository(session)
    if await repo.get(account.id) is None:
        await repo.upsert(account.id)  # create a default profile so the switch has a home
    await repo.set_kill_switch(account.id, body.on)
    return await repo.get(account.id)


@router.get("/accounts/{account_id}/positions", response_model=list[PositionOut])
async def list_positions(
    account: TradingAccountModel = Depends(get_owned_account),
    session: AsyncSession = Depends(get_db_session),
):
    return await PositionRepository(session).list_for_account(account.id)


@router.get("/accounts/{account_id}/orders", response_model=list[OrderOut])
async def list_orders(
    account: TradingAccountModel = Depends(get_owned_account),
    session: AsyncSession = Depends(get_db_session),
):
    return await OrderRepository(session).list_for_account(account.id)


@router.get("/accounts/{account_id}/trades", response_model=list[TradeOut])
async def list_trades(
    account: TradingAccountModel = Depends(get_owned_account),
    session: AsyncSession = Depends(get_db_session),
):
    return await TradeRepository(session).list_for_account(account.id)


@router.get("/accounts/{account_id}/equity", response_model=list[EquityPointOut])
async def equity_curve(
    account: TradingAccountModel = Depends(get_owned_account),
    session: AsyncSession = Depends(get_db_session),
):
    return await EquitySnapshotRepository(session).curve(account.id)
