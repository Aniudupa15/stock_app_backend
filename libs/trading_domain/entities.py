"""Immutable data contracts shared by every execution venue (paper, broker,
backtest) and the OMS/risk/strategy layers. See Phase 3 §1.

All money is `Decimal`; ids are `UUID`; times are timezone-aware IST (the
engine's convention - naive values are treated as IST wall-clock).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from libs.charges.models import Charges
from libs.trading_domain.enums import ExitReason, OrderState, OrderType, Product, Side, Validity


@dataclass(frozen=True, slots=True)
class MarketQuote:
    """The market snapshot a venue needs to fill against. `bid`/`ask`/
    `day_volume`/bands are optional - the paper venue degrades to `ltp` when
    depth isn't available (free NSE data doesn't always carry full depth)."""

    symbol: str
    ltp: Decimal
    bid: Decimal | None = None
    ask: Decimal | None = None
    day_volume: int | None = None
    upper_band: Decimal | None = None  # upper circuit
    lower_band: Decimal | None = None  # lower circuit
    ts: datetime | None = None


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """What the strategy/user WANTS - created and persisted before any venue
    call. `intent_id` is our idempotency key and the source of the broker
    order `tag` (Phase 3 §1)."""

    intent_id: UUID
    account_id: UUID
    symbol: str
    side: Side
    order_type: OrderType
    product: Product
    quantity: int
    price: Decimal | None = None  # LIMIT / SL
    trigger_price: Decimal | None = None  # SL / SL_M
    validity: Validity = Validity.DAY
    exchange: str = "NSE"
    strategy_id: UUID | None = None
    signal_id: UUID | None = None
    tag: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class OrderAck:
    """Synchronous response to place/cancel."""

    intent_id: UUID
    venue_order_id: str | None
    state: OrderState
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """Asynchronous push from a venue (fill / partial / reject / cancel)."""

    venue_order_id: str
    intent_id: UUID | None
    state: OrderState
    filled_qty: int
    pending_qty: int
    average_price: Decimal
    ts: datetime | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: str
    intent_id: UUID | None
    symbol: str
    side: Side
    qty: int
    price: Decimal
    charges: Charges
    ts: datetime | None = None


@dataclass(frozen=True, slots=True)
class Position:
    account_id: UUID
    symbol: str
    product: Product
    net_qty: int  # signed: +long / -short
    avg_price: Decimal
    realized_pnl: Decimal
    ltp: Decimal | None = None

    @property
    def unrealized_pnl(self) -> Decimal:
        if self.ltp is None or self.net_qty == 0:
            return Decimal("0")
        return (self.ltp - self.avg_price) * self.net_qty


@dataclass(frozen=True, slots=True)
class Holding:
    """A settled delivery (CNC) holding, as reported by a broker."""

    symbol: str
    quantity: int
    avg_price: Decimal
    ltp: Decimal | None = None


@dataclass(frozen=True, slots=True)
class Margin:
    available: Decimal
    required: Decimal = Decimal("0")

    @property
    def shortfall(self) -> Decimal:
        return max(Decimal("0"), self.required - self.available)


@dataclass(frozen=True, slots=True)
class TrailingSpec:
    """Trailing stop-loss config. `by`='PCT' -> distance = ltp*value/100;
    'POINTS' -> distance = value (absolute price). `step` is the minimum SL
    improvement before we bother modifying the resting SL order."""

    by: str  # "PCT" | "POINTS"
    value: Decimal
    step: Decimal = Decimal("0")


@dataclass(frozen=True, slots=True)
class BracketSpec:
    """Platform-managed bracket (Phase 3 §4.2). Single SL + single target in
    v1; multi-target partial exits are a documented later refinement. Since
    Zerodha deprecated native BO, this same spec drives paper AND live."""

    stop_loss: Decimal | None = None
    target: Decimal | None = None
    trailing: TrailingSpec | None = None


@dataclass(frozen=True, slots=True)
class Signal:
    strategy_id: UUID | None
    symbol: str
    side: Side
    entry: Decimal | None
    stop_loss: Decimal | None
    targets: list[Decimal] = field(default_factory=list)
    confidence: Decimal | None = None
    reasoning: str = ""
    ts: datetime | None = None


@dataclass(frozen=True, slots=True)
class Trade:
    """A closed round-trip, the unit of performance analytics."""

    account_id: UUID
    symbol: str
    qty: int
    entry_price: Decimal
    exit_price: Decimal
    pnl_gross: Decimal
    charges_total: Decimal
    pnl_net: Decimal
    entry_ts: datetime | None = None
    exit_ts: datetime | None = None
    strategy_id: UUID | None = None
    exit_reason: ExitReason | None = None

    @property
    def holding_seconds(self) -> int | None:
        if self.entry_ts is None or self.exit_ts is None:
            return None
        return int((self.exit_ts - self.entry_ts).total_seconds())
