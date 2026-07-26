"""Broker adapter interface - the multi-broker extensibility seam (Phase 1 §3).

Auth is expressed as a capability, not a fixed flow, because the five target
brokers diverge sharply: Zerodha = daily checksum login (token dies 6AM),
Angel One = TOTP, Upstox/Fyers = OAuth, Dhan = consent token. A concrete
adapter (ZerodhaAdapter first) implements this; `BrokerExecutionVenue` wraps
any adapter to satisfy `ExecutionVenuePort`, so the OMS/risk/strategy/engine
above it never learn which broker they're on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from libs.trading_domain.entities import Holding, Margin, OrderEvent, OrderIntent, Position


class BrokerError(Exception):
    """Any failure crossing the broker boundary (auth, network, rejection)."""


@dataclass(frozen=True, slots=True)
class BrokerSession:
    broker: str
    access_token: str
    expires_at: datetime | None = None  # Zerodha: ~06:00 IST next day

    def is_valid(self, now: datetime) -> bool:
        return self.expires_at is None or now < self.expires_at


class BrokerAdapter(ABC):
    # --- auth ---
    @abstractmethod
    def login_url(self) -> str:
        """URL to send the user to for interactive login."""

    @abstractmethod
    async def complete_login(self, callback: dict) -> BrokerSession:
        """Exchange the login callback (e.g. request_token) for a session."""

    @abstractmethod
    def set_session(self, session: BrokerSession) -> None: ...

    # --- trading ---
    @abstractmethod
    async def place_order(self, intent: OrderIntent) -> str:
        """Place an order; return the broker order id. Raises BrokerError."""

    @abstractmethod
    async def modify_order(self, order_id: str, intent: OrderIntent) -> str: ...

    @abstractmethod
    async def cancel_order(self, order_id: str, variety: str = "regular") -> str: ...

    @abstractmethod
    async def get_positions(self) -> list[Position]: ...

    @abstractmethod
    async def get_holdings(self) -> list[Holding]: ...

    @abstractmethod
    async def available_margin(self) -> Margin: ...

    @abstractmethod
    def parse_order_update(self, payload: dict, order_to_intent: dict[str, object]) -> OrderEvent:
        """Translate a broker order-update push into a normalised OrderEvent."""
