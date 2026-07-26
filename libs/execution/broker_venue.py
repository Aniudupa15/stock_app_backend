"""BrokerExecutionVenue - the third ExecutionVenue (Phase 2 P1).

Wraps any BrokerAdapter so the SAME OMS/risk/strategy/engine that were unit-
tested against PaperExecutionVenue now route real orders. The only behavioural
difference from paper: fills are confirmed asynchronously by the broker's
order stream, re-emitted through the event sink.

`handle_order_update(payload)` is called by the live order-update stream
(KiteTicker.on_order_update, hopped onto the event loop from its thread). In
tests it's called directly, so the whole venue is exercised without the
`kiteconnect` package or a network.
"""

from __future__ import annotations

from uuid import UUID

from libs.broker.base import BrokerAdapter, BrokerError
from libs.trading_domain.entities import Margin, OrderAck, OrderIntent, Position
from libs.trading_domain.enums import OrderState
from libs.trading_domain.ports import EventSink, ExecutionVenuePort


class BrokerExecutionVenue(ExecutionVenuePort):
    def __init__(self, adapter: BrokerAdapter) -> None:
        self._adapter = adapter
        self._event_sink: EventSink | None = None
        self._order_to_intent: dict[str, UUID] = {}

    def set_event_sink(self, sink: EventSink) -> None:
        self._event_sink = sink

    async def place(self, intent: OrderIntent) -> OrderAck:
        try:
            order_id = await self._adapter.place_order(intent)
        except BrokerError as exc:
            return OrderAck(intent.intent_id, None, OrderState.REJECTED, str(exc))
        self._order_to_intent[order_id] = intent.intent_id
        # SUBMITTED now; COMPLETE/REJECTED arrives via the order stream.
        return OrderAck(intent.intent_id, order_id, OrderState.SUBMITTED)

    async def cancel(self, venue_order_id: str) -> OrderAck:
        intent_id = self._order_to_intent.get(venue_order_id)
        try:
            await self._adapter.cancel_order(venue_order_id)
        except BrokerError as exc:
            return OrderAck(intent_id, venue_order_id, OrderState.REJECTED, str(exc))
        # Cancellation is confirmed by the order stream; report the request accepted.
        return OrderAck(intent_id, venue_order_id, OrderState.CANCELLED)

    async def positions(self) -> list[Position]:
        return await self._adapter.get_positions()

    async def available_margin(self) -> Margin:
        return await self._adapter.available_margin()

    def handle_order_update(self, payload: dict) -> None:
        """Entry point for the broker's async order-update push."""
        event = self._adapter.parse_order_update(payload, self._order_to_intent)
        if self._event_sink is not None:
            self._event_sink(event)
