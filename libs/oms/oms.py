"""Order Management System - platform-managed brackets, OCO, trailing SL.

Sits on top of any `ExecutionVenuePort` (paper today, Zerodha tomorrow) and
implements the bracket logic Zerodha no longer provides natively (BO
deprecated - Phase 1 §2.2), so the *same* code runs under paper and live.

Event handling is deferred, not reentrant: the venue's event sink only
*queues* events; `process()` acts on them outside any venue iteration, so
placing children / cancelling siblings never mutates a list the venue is
mid-loop over. The driving engine's tick cycle is:

    await venue.on_tick(quote)   # fills resting entries/children, queues events
    await oms.process()          # entry filled -> place children; child filled -> OCO + Trade
    await oms.on_tick(quote)     # trailing-SL adjustments

Risk is intentionally NOT here - the engine runs the risk gate before calling
`submit()`, keeping this component purely about order orchestration.

v1 scope (documented): one SL + one target per bracket; MARKET or resting
entry; trailing by PCT/POINTS. Multi-target partial exits and ATR-trailing
are later refinements.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from libs.charges.nse_equity import compute
from libs.trading_domain.entities import (
    BracketSpec,
    OrderAck,
    OrderEvent,
    OrderIntent,
    Trade,
)
from libs.trading_domain.enums import ExitReason, OrderState, OrderType, Side
from libs.trading_domain.ports import ExecutionVenuePort


@dataclass
class _Bracket:
    bracket_id: str
    entry_intent: OrderIntent
    spec: BracketSpec
    direction: int  # +1 long, -1 short
    state: str = "PENDING_ENTRY"  # PENDING_ENTRY | ACTIVE | CLOSED
    entry_order_id: str | None = None
    entry_price: Decimal | None = None
    entry_qty: int = 0
    entry_ts: datetime | None = None
    sl_order_id: str | None = None
    target_order_id: str | None = None
    current_sl: Decimal | None = None


@dataclass
class OMSResult:
    accepted: bool
    bracket_id: str | None
    ack: OrderAck | None
    reason: str | None = None


class OrderManagementSystem:
    def __init__(self, venue: ExecutionVenuePort, clock):
        self._venue = venue
        self._clock = clock
        self._brackets: dict[str, _Bracket] = {}
        self._order_to_bracket: dict[str, str] = {}
        self._events: list[OrderEvent] = []
        self._counter = 0
        self.trades: list[Trade] = []
        venue.set_event_sink(self._events.append)

    async def submit(self, intent: OrderIntent, spec: BracketSpec | None = None) -> OMSResult:
        """Place a (risk-approved) entry and register its bracket. Call
        `process()` afterwards to place the SL/target once the entry fills."""
        spec = spec or BracketSpec()
        direction = 1 if intent.side is Side.BUY else -1
        ack = await self._venue.place(intent)
        if ack.state is OrderState.REJECTED:
            return OMSResult(False, None, ack, ack.reason)

        bracket_id = self._next_id("BRK")
        bracket = _Bracket(bracket_id, intent, spec, direction, entry_order_id=ack.venue_order_id)
        self._brackets[bracket_id] = bracket
        if ack.venue_order_id:
            self._order_to_bracket[ack.venue_order_id] = bracket_id
        return OMSResult(True, bracket_id, ack)

    async def process(self) -> None:
        """Drain queued venue events and act on them."""
        while self._events:
            event = self._events.pop(0)
            await self._handle(event)

    async def on_tick(self, quote) -> None:
        """Trailing-SL maintenance for active brackets on this symbol."""
        for bracket in list(self._brackets.values()):
            if bracket.state != "ACTIVE" or bracket.spec.trailing is None:
                continue
            if bracket.entry_intent.symbol != quote.symbol:
                continue
            await self._maybe_trail(bracket, quote.ltp)

    # --- event handling ---

    async def _handle(self, event: OrderEvent) -> None:
        bracket_id = self._order_to_bracket.get(event.venue_order_id)
        if bracket_id is None:
            return
        bracket = self._brackets[bracket_id]

        if event.venue_order_id == bracket.entry_order_id:
            if event.state is OrderState.COMPLETE and bracket.state == "PENDING_ENTRY":
                await self._activate(bracket, event)
            return

        # A child (SL or target) filled -> OCO + record the trade.
        if event.state is OrderState.COMPLETE and bracket.state == "ACTIVE":
            await self._close(bracket, event)

    async def _activate(self, bracket: _Bracket, entry_event: OrderEvent) -> None:
        bracket.entry_price = entry_event.average_price
        bracket.entry_qty = entry_event.filled_qty
        bracket.entry_ts = entry_event.ts
        bracket.state = "ACTIVE"

        exit_side = Side.SELL if bracket.direction == 1 else Side.BUY
        if bracket.spec.stop_loss is not None:
            bracket.current_sl = bracket.spec.stop_loss
            bracket.sl_order_id = await self._place_child(bracket, exit_side, OrderType.SL_M, trigger=bracket.spec.stop_loss)
        if bracket.spec.target is not None:
            bracket.target_order_id = await self._place_child(
                bracket, exit_side, OrderType.LIMIT, price=bracket.spec.target
            )

    async def _close(self, bracket: _Bracket, exit_event: OrderEvent) -> None:
        # Cancel the sibling (OCO).
        filled_id = exit_event.venue_order_id
        sibling = bracket.target_order_id if filled_id == bracket.sl_order_id else bracket.sl_order_id
        exit_reason = ExitReason.STOP_LOSS if filled_id == bracket.sl_order_id else ExitReason.TARGET
        if sibling is not None:
            await self._venue.cancel(sibling)

        bracket.state = "CLOSED"
        self.trades.append(self._build_trade(bracket, exit_event, exit_reason))

    def _build_trade(self, bracket: _Bracket, exit_event: OrderEvent, reason: ExitReason) -> Trade:
        entry = bracket.entry_price or Decimal("0")
        exit_price = exit_event.average_price
        qty = bracket.entry_qty
        pnl_gross = (exit_price - entry) * qty * bracket.direction
        entry_side = bracket.entry_intent.side
        exit_side = Side.SELL if entry_side is Side.BUY else Side.BUY
        product = bracket.entry_intent.product
        charges_total = self._leg_charges(bracket.entry_order_id, entry_side, product, qty, entry) + self._leg_charges(
            exit_event.venue_order_id, exit_side, product, qty, exit_price
        )
        return Trade(
            account_id=bracket.entry_intent.account_id,
            symbol=bracket.entry_intent.symbol,
            qty=qty,
            entry_price=entry,
            exit_price=exit_price,
            pnl_gross=pnl_gross,
            charges_total=charges_total,
            pnl_net=pnl_gross - charges_total,
            entry_ts=bracket.entry_ts,
            exit_ts=exit_event.ts,
            strategy_id=bracket.entry_intent.strategy_id,
            exit_reason=reason,
        )

    async def _maybe_trail(self, bracket: _Bracket, ltp: Decimal) -> None:
        spec = bracket.spec.trailing
        distance = ltp * spec.value / Decimal("100") if spec.by == "PCT" else spec.value
        if bracket.direction == 1:
            candidate = ltp - distance
            improves = bracket.current_sl is None or candidate - bracket.current_sl >= spec.step
            better = bracket.current_sl is None or candidate > bracket.current_sl
        else:
            candidate = ltp + distance
            improves = bracket.current_sl is None or bracket.current_sl - candidate >= spec.step
            better = bracket.current_sl is None or candidate < bracket.current_sl
        if not (improves and better):
            return
        # Cancel + replace the SL child at the tightened level.
        if bracket.sl_order_id is not None:
            await self._venue.cancel(bracket.sl_order_id)
            self._order_to_bracket.pop(bracket.sl_order_id, None)
        exit_side = Side.SELL if bracket.direction == 1 else Side.BUY
        bracket.current_sl = candidate
        bracket.sl_order_id = await self._place_child(bracket, exit_side, OrderType.SL_M, trigger=candidate)

    async def _place_child(
        self,
        bracket: _Bracket,
        side: Side,
        order_type: OrderType,
        *,
        price: Decimal | None = None,
        trigger: Decimal | None = None,
    ) -> str | None:
        child = replace(
            bracket.entry_intent,
            intent_id=uuid4(),
            side=side,
            order_type=order_type,
            price=price,
            trigger_price=trigger,
            quantity=bracket.entry_qty,
        )
        ack = await self._venue.place(child)
        if ack.venue_order_id:
            self._order_to_bracket[ack.venue_order_id] = bracket.bracket_id
        return ack.venue_order_id

    def _leg_charges(self, order_id: str | None, side, product, qty: int, price: Decimal) -> Decimal:
        # Prefer the venue's exact recorded charges (paper); fall back to
        # computing them (broker path - live actuals reconciled from the
        # contract note later).
        get_fill = getattr(self._venue, "get_fill", None)
        if order_id is not None and get_fill is not None:
            fill = get_fill(order_id)
            if fill is not None:
                return fill.charges.total
        if qty <= 0 or price <= 0:
            return Decimal("0")
        return compute(side, product, qty, price).total

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:06d}"

    # introspection for tests / engine
    def bracket_state(self, bracket_id: str) -> str:
        return self._brackets[bracket_id].state
