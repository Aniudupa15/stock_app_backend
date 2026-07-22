import uuid
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from app.domain.entities import TransactionType


class PortfolioCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)


class PortfolioOut(BaseModel):
    id: uuid.UUID
    name: str
    created_at: datetime


class TransactionCreate(BaseModel):
    symbol: str = Field(..., min_length=1)
    transaction_type: TransactionType
    quantity: Decimal = Field(..., gt=0)
    price: Decimal = Field(..., gt=0)
    transaction_date: date


class HoldingOut(BaseModel):
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    cost_basis: Decimal
    current_price: Decimal | None
    current_value: Decimal | None
    pnl: Decimal | None
    pnl_percent: Decimal | None


class PortfolioDetailOut(BaseModel):
    id: uuid.UUID
    name: str
    created_at: datetime
    holdings: list[HoldingOut]


class PortfolioPerformanceOut(BaseModel):
    id: uuid.UUID
    total_invested: Decimal
    current_value: Decimal
    total_pnl: Decimal
    total_pnl_percent: Decimal | None
    xirr_percent: Decimal | None
