"""SQLAlchemy models for the trading platform (Phase 3 §2).

All 16 tables live in a dedicated `trading` Postgres schema, sharing the
data-service's declarative `Base` (one metadata -> one Alembic chain). FKs to
the shared `users`/`stocks` tables reference the default (public) schema;
FKs between trading tables are schema-qualified (`trading.<table>.id`).

Enum-like fields are stored as short strings (matching the existing app's
convention) rather than native PG enums, to keep migrations simple.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin

SCHEMA = "trading"


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))


class BrokerSessionModel(Base, TimestampMixin):
    __tablename__ = "broker_sessions"
    __table_args__ = (UniqueConstraint("user_id", "broker", name="uq_broker_sessions_user_broker"), {"schema": SCHEMA})

    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    broker: Mapped[str] = mapped_column(String(32), nullable=False)
    api_key_enc: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    api_secret_enc: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    access_token_enc: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    access_token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="DISCONNECTED")


class TradingAccountModel(Base, TimestampMixin):
    __tablename__ = "trading_accounts"
    __table_args__ = ({"schema": SCHEMA},)

    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    mode: Mapped[str] = mapped_column(String(8), nullable=False, server_default="PAPER")
    broker_session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.broker_sessions.id", ondelete="SET NULL"), nullable=True
    )
    virtual_balance: Mapped[float | None] = mapped_column(Numeric(18, 2), nullable=True)
    starting_balance: Mapped[float | None] = mapped_column(Numeric(18, 2), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))


class StrategyModel(Base, TimestampMixin):
    __tablename__ = "strategies"
    __table_args__ = ({"schema": SCHEMA},)

    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    rule_tree: Mapped[dict] = mapped_column(JSONB, nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, server_default="1D")
    universe: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    side: Mapped[str] = mapped_column(String(4), nullable=False, server_default="BUY")
    product: Mapped[str] = mapped_column(String(4), nullable=False, server_default="MIS")
    exit_rule: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    status: Mapped[str] = mapped_column(String(12), nullable=False, server_default="DRAFT")
    validation: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class OrderIntentModel(Base):
    __tablename__ = "order_intents"
    __table_args__ = ({"schema": SCHEMA},)

    id: Mapped[uuid.UUID] = _pk()
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.trading_accounts.id", ondelete="CASCADE"), nullable=False
    )
    strategy_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.strategies.id", ondelete="SET NULL"), nullable=True
    )
    signal_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    order_type: Mapped[str] = mapped_column(String(8), nullable=False)
    product: Mapped[str] = mapped_column(String(4), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[float | None] = mapped_column(Numeric(18, 4), nullable=True)
    trigger_price: Mapped[float | None] = mapped_column(Numeric(18, 4), nullable=True)
    validity: Mapped[str] = mapped_column(String(4), nullable=False, server_default="DAY")
    bracket: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    risk_verdict: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class OrderModel(Base, TimestampMixin):
    __tablename__ = "orders"
    __table_args__ = (UniqueConstraint("venue", "venue_order_id", name="uq_orders_venue_order"), {"schema": SCHEMA})

    id: Mapped[uuid.UUID] = _pk()
    intent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.order_intents.id", ondelete="CASCADE"), nullable=False
    )
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.trading_accounts.id", ondelete="CASCADE"), nullable=False
    )
    venue: Mapped[str] = mapped_column(String(16), nullable=False)
    venue_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    exchange_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parent_order_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.orders.id", ondelete="SET NULL"), nullable=True
    )
    leg: Mapped[str | None] = mapped_column(String(8), nullable=True)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    filled_qty: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    pending_qty: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    average_price: Mapped[float | None] = mapped_column(Numeric(18, 4), nullable=True)
    reject_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class FillModel(Base):
    __tablename__ = "fills"
    __table_args__ = ({"schema": SCHEMA},)

    id: Mapped[uuid.UUID] = _pk()
    order_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.orders.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    charges: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class BracketGroupModel(Base, TimestampMixin):
    __tablename__ = "bracket_groups"
    __table_args__ = ({"schema": SCHEMA},)

    id: Mapped[uuid.UUID] = _pk()
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.trading_accounts.id", ondelete="CASCADE"), nullable=False
    )
    entry_order_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    sl_order_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    target_order_ids: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    trailing: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    state: Mapped[str] = mapped_column(String(24), nullable=False, server_default="OPEN")


class PositionModel(Base, TimestampMixin):
    __tablename__ = "positions"
    __table_args__ = (
        UniqueConstraint("account_id", "symbol", "product", name="uq_positions_account_symbol_product"),
        {"schema": SCHEMA},
    )

    id: Mapped[uuid.UUID] = _pk()
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.trading_accounts.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    product: Mapped[str] = mapped_column(String(4), nullable=False)
    net_qty: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    avg_price: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, server_default="0")
    realized_pnl: Mapped[float] = mapped_column(Numeric(18, 2), nullable=False, server_default="0")
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class HoldingModel(Base, TimestampMixin):
    __tablename__ = "holdings"
    __table_args__ = (
        UniqueConstraint("account_id", "symbol", name="uq_holdings_account_symbol"),
        {"schema": SCHEMA},
    )

    id: Mapped[uuid.UUID] = _pk()
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.trading_accounts.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    avg_price: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, server_default="0")
    ltp: Mapped[float | None] = mapped_column(Numeric(18, 4), nullable=True)


class TradeModel(Base):
    __tablename__ = "trades"
    __table_args__ = ({"schema": SCHEMA},)

    id: Mapped[uuid.UUID] = _pk()
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.trading_accounts.id", ondelete="CASCADE"), nullable=False
    )
    strategy_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.strategies.id", ondelete="SET NULL"), nullable=True
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    entry_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_price: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    exit_price: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    pnl_gross: Mapped[float] = mapped_column(Numeric(18, 2), nullable=False)
    charges_total: Mapped[float] = mapped_column(Numeric(18, 2), nullable=False, server_default="0")
    pnl_net: Mapped[float] = mapped_column(Numeric(18, 2), nullable=False)
    r_multiple: Mapped[float | None] = mapped_column(Numeric(10, 3), nullable=True)
    holding_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class RiskProfileModel(Base, TimestampMixin):
    __tablename__ = "risk_profiles"
    __table_args__ = (UniqueConstraint("account_id", name="uq_risk_profiles_account"), {"schema": SCHEMA})

    id: Mapped[uuid.UUID] = _pk()
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.trading_accounts.id", ondelete="CASCADE"), nullable=False
    )
    max_daily_loss: Mapped[float | None] = mapped_column(Numeric(18, 2), nullable=True)
    max_weekly_loss: Mapped[float | None] = mapped_column(Numeric(18, 2), nullable=True)
    max_monthly_loss: Mapped[float | None] = mapped_column(Numeric(18, 2), nullable=True)
    max_capital_alloc_pct: Mapped[float | None] = mapped_column(Numeric(6, 3), nullable=True)
    per_trade_risk_pct: Mapped[float | None] = mapped_column(Numeric(6, 3), nullable=True)
    max_open_positions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_exposure: Mapped[float | None] = mapped_column(Numeric(18, 2), nullable=True)
    cooldown_losses: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cooldown_minutes: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    square_off_buffer_minutes: Mapped[int] = mapped_column(Integer, nullable=False, server_default="15")
    kill_switch: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))


class RiskEventModel(Base):
    __tablename__ = "risk_events"
    __table_args__ = ({"schema": SCHEMA},)

    id: Mapped[uuid.UUID] = _pk()
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.trading_accounts.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    detail: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class EquitySnapshotModel(Base):
    __tablename__ = "equity_snapshots"
    __table_args__ = ({"schema": SCHEMA},)

    id: Mapped[uuid.UUID] = _pk()
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.trading_accounts.id", ondelete="CASCADE"), nullable=False
    )
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    equity: Mapped[float] = mapped_column(Numeric(18, 2), nullable=False)
    cash: Mapped[float] = mapped_column(Numeric(18, 2), nullable=False)
    unrealized: Mapped[float | None] = mapped_column(Numeric(18, 2), nullable=True)


class SignalModel(Base):
    __tablename__ = "signals"
    __table_args__ = ({"schema": SCHEMA},)

    id: Mapped[uuid.UUID] = _pk()
    strategy_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.strategies.id", ondelete="SET NULL"), nullable=True
    )
    account_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.trading_accounts.id", ondelete="CASCADE"), nullable=True
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    entry: Mapped[float | None] = mapped_column(Numeric(18, 4), nullable=True)
    stop_loss: Mapped[float | None] = mapped_column(Numeric(18, 4), nullable=True)
    targets: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    acted: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AuditLogModel(Base):
    __tablename__ = "audit_log"
    __table_args__ = ({"schema": SCHEMA},)

    id: Mapped[uuid.UUID] = _pk()
    account_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.trading_accounts.id", ondelete="CASCADE"), nullable=True
    )
    actor: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(48), nullable=False)
    ref_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class BacktestModel(Base):
    __tablename__ = "backtests"
    __table_args__ = ({"schema": SCHEMA},)

    id: Mapped[uuid.UUID] = _pk()
    account_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.trading_accounts.id", ondelete="SET NULL"), nullable=True
    )
    strategy_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.strategies.id", ondelete="SET NULL"), nullable=True
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    from_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    to_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    starting_cash: Mapped[float] = mapped_column(Numeric(18, 2), nullable=False)
    final_equity: Mapped[float] = mapped_column(Numeric(18, 2), nullable=False)
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


ALL_MODELS = [
    BrokerSessionModel,
    TradingAccountModel,
    StrategyModel,
    OrderIntentModel,
    OrderModel,
    FillModel,
    BracketGroupModel,
    PositionModel,
    HoldingModel,
    TradeModel,
    RiskProfileModel,
    RiskEventModel,
    EquitySnapshotModel,
    SignalModel,
    AuditLogModel,
    BacktestModel,
]
