"""Ports the trading engine depends on. Concrete venues (paper, Zerodha,
backtest) implement `ExecutionVenuePort`; the data-service (or a fake, in
tests) implements `QuoteSourcePort`.

Order events are delivered via a registered sink callback rather than an
async iterator: it composes cleanly for both the synchronous paper venue and
a live adapter that pushes from a WebSocket thread (via
`loop.call_soon_threadsafe`). The OMS registers the sink.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from libs.trading_domain.entities import Margin, MarketQuote, OrderAck, OrderEvent, OrderIntent, Position

EventSink = Callable[[OrderEvent], None]


class QuoteSourcePort(ABC):
    @abstractmethod
    async def get_quote(self, symbol: str) -> MarketQuote:
        """Latest market snapshot for a symbol. Raises on unavailability -
        the caller decides whether that blocks a fill."""


class ExecutionVenuePort(ABC):
    """One interface, three implementations (paper/broker/backtest). Strategy,
    OMS, and risk code talk only to this - that is what makes paper->live a
    configuration change (Phase 2 P1)."""

    @abstractmethod
    async def place(self, intent: OrderIntent) -> OrderAck: ...

    @abstractmethod
    async def cancel(self, venue_order_id: str) -> OrderAck: ...

    @abstractmethod
    async def positions(self) -> list[Position]: ...

    @abstractmethod
    async def available_margin(self) -> Margin: ...

    @abstractmethod
    def set_event_sink(self, sink: EventSink) -> None:
        """Register the callback that receives async order events."""
