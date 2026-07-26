"""Strategy CRUD endpoints (white-box rule trees)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from libs.strategy.dsl import RuleError, evaluate_rule
from services.trading_service.api.deps import get_current_user_id, get_db_session
from services.trading_service.api.schemas import StrategyCreate, StrategyOut, StrategyUpdate
from services.trading_service.persistence.models import StrategyModel
from services.trading_service.persistence.repositories import StrategyRepository

router = APIRouter(prefix="/trading/strategies", tags=["strategies"])


def _validate_rule(rule_tree: dict) -> None:
    # Structural validation only: run the rule against an empty feature set. A
    # missing feature is legal (reads False); a malformed tree raises RuleError.
    try:
        evaluate_rule(rule_tree, {}, None)
    except RuleError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"invalid rule: {exc}") from exc


@router.post("", response_model=StrategyOut, status_code=status.HTTP_201_CREATED)
async def create_strategy(
    body: StrategyCreate,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
) -> StrategyModel:
    _validate_rule(body.rule_tree)
    exit_rule = {}
    if body.stop_loss_pct is not None:
        exit_rule["stop_loss_pct"] = str(body.stop_loss_pct)
    if body.target_pct is not None:
        exit_rule["target_pct"] = str(body.target_pct)
    return await StrategyRepository(session).create(
        user_id,
        name=body.name,
        rule_tree=body.rule_tree,
        side=body.side,
        product=body.product,
        quantity=body.quantity,
        timeframe=body.timeframe,
        exit_rule=exit_rule or None,
    )


@router.get("", response_model=list[StrategyOut])
async def list_strategies(
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
) -> list[StrategyModel]:
    return await StrategyRepository(session).list_for_user(user_id)


@router.get("/{strategy_id}", response_model=StrategyOut)
async def get_strategy(
    strategy_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
) -> StrategyModel:
    strategy = await StrategyRepository(session).get(strategy_id, user_id)
    if strategy is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="strategy not found")
    return strategy


@router.patch("/{strategy_id}", response_model=StrategyOut)
async def update_strategy(
    strategy_id: uuid.UUID,
    body: StrategyUpdate,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
) -> StrategyModel:
    fields = body.model_dump(exclude_none=True)
    if "rule_tree" in fields:
        _validate_rule(fields["rule_tree"])
    repo = StrategyRepository(session)
    if fields and not await repo.update(strategy_id, user_id, **fields):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="strategy not found")
    strategy = await repo.get(strategy_id, user_id)
    if strategy is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="strategy not found")
    return strategy


@router.delete("/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_strategy(
    strategy_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
):
    if not await StrategyRepository(session).delete(strategy_id, user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="strategy not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
