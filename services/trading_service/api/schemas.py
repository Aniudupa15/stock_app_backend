"""Pydantic request/response models for the trading service API."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AccountCreate(BaseModel):
    mode: str = Field(default="PAPER", pattern="^(PAPER|LIVE)$")
    starting_balance: Decimal = Field(default=Decimal("10000"), gt=0)


class AccountOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    mode: str
    virtual_balance: Decimal | None
    starting_balance: Decimal | None
    is_active: bool
    created_at: datetime


class RiskProfileIn(BaseModel):
    max_daily_loss: Decimal | None = None
    max_weekly_loss: Decimal | None = None
    max_monthly_loss: Decimal | None = None
    per_trade_risk_pct: Decimal | None = None
    max_open_positions: int | None = None
    max_exposure: Decimal | None = None
    cooldown_losses: int | None = None
    cooldown_minutes: int | None = None
    square_off_buffer_minutes: int | None = None


class RiskProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    account_id: uuid.UUID
    max_daily_loss: Decimal | None
    max_weekly_loss: Decimal | None
    max_monthly_loss: Decimal | None
    per_trade_risk_pct: Decimal | None
    max_open_positions: int | None
    max_exposure: Decimal | None
    cooldown_losses: int | None
    cooldown_minutes: int
    square_off_buffer_minutes: int
    kill_switch: bool


class KillSwitchIn(BaseModel):
    on: bool


class StrategyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    rule_tree: dict
    side: str = Field(default="BUY", pattern="^(BUY|SELL)$")
    product: str = Field(default="MIS", pattern="^(MIS|CNC)$")
    quantity: int = Field(default=1, gt=0)
    timeframe: str = "1D"
    stop_loss_pct: Decimal | None = None
    target_pct: Decimal | None = None


class StrategyUpdate(BaseModel):
    name: str | None = None
    status: str | None = None
    quantity: int | None = Field(default=None, gt=0)
    rule_tree: dict | None = None


class StrategyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    name: str
    rule_tree: dict
    side: str
    product: str
    quantity: int
    status: str
    timeframe: str
    exit_rule: dict | None
    created_at: datetime


class BacktestRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)
    strategy_id: uuid.UUID | None = None
    # inline strategy (used when strategy_id is absent)
    rule_tree: dict | None = None
    side: str = Field(default="BUY", pattern="^(BUY|SELL)$")
    product: str = Field(default="CNC", pattern="^(MIS|CNC)$")
    quantity: int = Field(default=1, gt=0)
    stop_loss_pct: Decimal | None = None
    target_pct: Decimal | None = None
    from_date: date | None = None
    to_date: date | None = None
    starting_cash: Decimal = Field(default=Decimal("1000000"), gt=0)


class BacktestOut(BaseModel):
    id: uuid.UUID
    symbol: str
    starting_cash: Decimal
    final_equity: Decimal
    bars: int
    equity_points: int
    metrics: dict[str, Any]


class BrokerConnectIn(BaseModel):
    broker: str = Field(default="zerodha", pattern="^zerodha$")
    api_key: str = Field(min_length=1, max_length=64)
    api_secret: str = Field(min_length=1, max_length=128)


class BrokerConnectOut(BaseModel):
    broker: str
    login_url: str
    status: str


class BrokerCompleteIn(BaseModel):
    broker: str = Field(default="zerodha", pattern="^zerodha$")
    request_token: str = Field(min_length=1)


class BrokerStatusOut(BaseModel):
    broker: str
    connected: bool
    status: str
    expires_at: datetime | None = None


class PaperRunRequest(BaseModel):
    """Run a saved strategy over a symbol's history as a persisted paper
    session (trades + equity curve committed to the account)."""

    strategy_id: uuid.UUID
    symbol: str = Field(min_length=1, max_length=32)
    from_date: date | None = None
    to_date: date | None = None


class PaperRunOut(BaseModel):
    account_id: uuid.UUID
    symbol: str
    bars: int
    trades: int
    net_pnl: Decimal
    final_equity: Decimal
    metrics: dict[str, Any]


class PositionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    symbol: str
    product: str
    net_qty: int
    avg_price: Decimal
    realized_pnl: Decimal


class OrderOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    venue: str
    venue_order_id: str | None
    state: str
    leg: str | None
    filled_qty: int
    pending_qty: int
    average_price: Decimal | None
    created_at: datetime


class EquityPointOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    ts: datetime
    equity: Decimal
    cash: Decimal


class TradeOut(BaseModel):
    symbol: str
    qty: int
    entry_price: Decimal
    exit_price: Decimal
    pnl_gross: Decimal
    charges_total: Decimal
    pnl_net: Decimal
    exit_reason: str | None
    entry_ts: datetime | None
    exit_ts: datetime | None
