import uuid

from fastapi import APIRouter, Depends, status

from app.api.deps import get_current_user_id, get_portfolio_service
from app.schemas.portfolio import (
    PortfolioCreate,
    PortfolioDetailOut,
    PortfolioOut,
    PortfolioPerformanceOut,
    TransactionCreate,
)
from app.services.portfolio_service import PortfolioService

router = APIRouter(prefix="/portfolios", tags=["portfolios"])


@router.post("", response_model=PortfolioOut, status_code=status.HTTP_201_CREATED)
async def create_portfolio(
    body: PortfolioCreate,
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> PortfolioOut:
    return await service.create(user_id, body.name)


@router.get("", response_model=list[PortfolioOut])
async def list_portfolios(
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> list[PortfolioOut]:
    return await service.list(user_id)


@router.get("/{portfolio_id}", response_model=PortfolioDetailOut)
async def get_portfolio(
    portfolio_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> PortfolioDetailOut:
    return await service.get_detail(user_id, portfolio_id)


@router.post("/{portfolio_id}/transactions", response_model=PortfolioDetailOut)
async def add_transaction(
    portfolio_id: uuid.UUID,
    body: TransactionCreate,
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> PortfolioDetailOut:
    return await service.add_transaction(user_id, portfolio_id, body)


@router.get("/{portfolio_id}/performance", response_model=PortfolioPerformanceOut)
async def get_portfolio_performance(
    portfolio_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> PortfolioPerformanceOut:
    return await service.get_performance(user_id, portfolio_id)
