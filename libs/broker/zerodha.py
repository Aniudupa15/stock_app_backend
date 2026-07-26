"""Zerodha Kite Connect adapter (verified against kite.trade/docs/connect/v3).

Wraps a Kite client that is *injected*, so the pure mapping logic (intent ->
Kite params, Kite status -> normalised state, order-update -> OrderEvent) is
unit-tested with a fake and needs neither the `kiteconnect` package nor
credentials. `create_live_adapter()` lazily imports the real KiteConnect for
production use.

Kite calls are synchronous/blocking HTTP; they run in a thread via
`asyncio.to_thread` so they never block the engine's event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, time, timedelta
from decimal import Decimal
from uuid import UUID

from libs.broker.base import BrokerAdapter, BrokerError, BrokerSession
from libs.trading_calendar.calendar import IST
from libs.trading_domain.entities import Holding, Margin, OrderEvent, OrderIntent, Position
from libs.trading_domain.enums import OrderState, OrderType, Product, Side

_ORDER_TYPE_TO_KITE = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.SL: "SL",
    OrderType.SL_M: "SL-M",
}

# Kite order statuses (Phase 1 §2.2) -> our normalised lifecycle.
_STATUS_TO_STATE = {
    "COMPLETE": OrderState.COMPLETE,
    "REJECTED": OrderState.REJECTED,
    "CANCELLED": OrderState.CANCELLED,
    "OPEN": OrderState.OPEN,
    "TRIGGER PENDING": OrderState.OPEN,
    "PUT ORDER REQ RECEIVED": OrderState.SUBMITTED,
    "VALIDATION PENDING": OrderState.SUBMITTED,
    "OPEN PENDING": OrderState.SUBMITTED,
    "MODIFY VALIDATION PENDING": OrderState.OPEN,
    "MODIFY PENDING": OrderState.OPEN,
    "AMO REQ RECEIVED": OrderState.SUBMITTED,
    "CANCEL PENDING": OrderState.OPEN,
}


def intent_to_kite_params(intent: OrderIntent) -> dict:
    """Map an OrderIntent to Kite place_order kwargs. Pure and testable."""
    params: dict = {
        "variety": "regular",
        "exchange": intent.exchange,
        "tradingsymbol": intent.symbol,
        "transaction_type": "BUY" if intent.side is Side.BUY else "SELL",
        "quantity": intent.quantity,
        "product": intent.product.value,  # MIS / CNC (1:1 with Kite)
        "order_type": _ORDER_TYPE_TO_KITE[intent.order_type],
        "validity": intent.validity.value,
        "tag": intent.tag or intent.intent_id.hex[:20],  # Kite tag max 20 chars
    }
    if intent.price is not None and intent.order_type in (OrderType.LIMIT, OrderType.SL):
        params["price"] = float(intent.price)
    if intent.trigger_price is not None and intent.order_type in (OrderType.SL, OrderType.SL_M):
        params["trigger_price"] = float(intent.trigger_price)
    return params


def kite_status_to_state(status: str, filled_qty: int, pending_qty: int) -> OrderState:
    base = _STATUS_TO_STATE.get(status.upper(), OrderState.SUBMITTED)
    if base is OrderState.OPEN and filled_qty > 0 and pending_qty > 0:
        return OrderState.PARTIAL
    if base is OrderState.OPEN and filled_qty > 0 and pending_qty == 0:
        return OrderState.COMPLETE
    return base


class ZerodhaAdapter(BrokerAdapter):
    def __init__(
        self,
        kite,
        api_key: str,
        api_secret: str,
        account_id: UUID,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._kite = kite
        self._api_key = api_key
        self._api_secret = api_secret
        self._account_id = account_id
        self._clock = clock or (lambda: datetime.now(IST))

    # --- auth ---

    def login_url(self) -> str:
        return f"https://kite.zerodha.com/connect/login?v=3&api_key={self._api_key}"

    async def complete_login(self, callback: dict) -> BrokerSession:
        request_token = callback["request_token"]
        try:
            data = await asyncio.to_thread(self._kite.generate_session, request_token, self._api_secret)
        except Exception as exc:  # external boundary - normalise the error
            raise BrokerError(f"Kite login failed: {exc}") from exc
        access_token = data["access_token"]
        self._kite.set_access_token(access_token)
        return BrokerSession("zerodha", access_token, self._next_expiry())

    def set_session(self, session: BrokerSession) -> None:
        self._kite.set_access_token(session.access_token)

    def _next_expiry(self) -> datetime:
        # Kite access token expires ~06:00 IST the next day (regulatory).
        now = self._clock()
        today_6am = datetime.combine(now.date(), time(6, 0), tzinfo=now.tzinfo)
        return today_6am if now < today_6am else today_6am + timedelta(days=1)

    # --- trading ---

    async def place_order(self, intent: OrderIntent) -> str:
        try:
            return await asyncio.to_thread(self._kite.place_order, **intent_to_kite_params(intent))
        except Exception as exc:
            raise BrokerError(f"place_order failed: {exc}") from exc

    async def modify_order(self, order_id: str, intent: OrderIntent) -> str:
        params = intent_to_kite_params(intent)
        params["order_id"] = order_id
        try:
            return await asyncio.to_thread(self._kite.modify_order, **params)
        except Exception as exc:
            raise BrokerError(f"modify_order failed: {exc}") from exc

    async def cancel_order(self, order_id: str, variety: str = "regular") -> str:
        try:
            return await asyncio.to_thread(self._kite.cancel_order, variety, order_id)
        except Exception as exc:
            raise BrokerError(f"cancel_order failed: {exc}") from exc

    async def get_positions(self) -> list[Position]:
        raw = await asyncio.to_thread(self._kite.positions)
        out: list[Position] = []
        for p in raw.get("net", []):
            out.append(
                Position(
                    account_id=self._account_id,
                    symbol=p["tradingsymbol"],
                    product=Product(p["product"]) if p.get("product") in ("MIS", "CNC") else Product.MIS,
                    net_qty=int(p["quantity"]),
                    avg_price=Decimal(str(p.get("average_price", 0))),
                    realized_pnl=Decimal(str(p.get("realised", 0))),
                    ltp=Decimal(str(p["last_price"])) if p.get("last_price") is not None else None,
                )
            )
        return out

    async def get_holdings(self) -> list[Holding]:
        raw = await asyncio.to_thread(self._kite.holdings)
        return [
            Holding(
                symbol=h["tradingsymbol"],
                quantity=int(h["quantity"]),
                avg_price=Decimal(str(h.get("average_price", 0))),
                ltp=Decimal(str(h["last_price"])) if h.get("last_price") is not None else None,
            )
            for h in raw
        ]

    async def available_margin(self) -> Margin:
        raw = await asyncio.to_thread(self._kite.margins, "equity")
        available = raw.get("available", {})
        cash = available.get("live_balance", available.get("cash", 0))
        return Margin(available=Decimal(str(cash)))

    def parse_order_update(self, payload: dict, order_to_intent: dict[str, object]) -> OrderEvent:
        order_id = payload["order_id"]
        filled = int(payload.get("filled_quantity", 0) or 0)
        pending = int(payload.get("pending_quantity", 0) or 0)
        state = kite_status_to_state(payload.get("status", ""), filled, pending)
        intent_id = order_to_intent.get(order_id)
        return OrderEvent(
            venue_order_id=order_id,
            intent_id=intent_id,  # type: ignore[arg-type]
            state=state,
            filled_qty=filled,
            pending_qty=pending,
            average_price=Decimal(str(payload.get("average_price", 0) or 0)),
            reason=payload.get("status_message"),
        )


def create_live_adapter(api_key: str, api_secret: str, account_id: UUID, access_token: str | None = None):
    """Build a ZerodhaAdapter around the real KiteConnect client. Imports
    `kiteconnect` lazily so the rest of the platform never depends on it."""
    try:
        from kiteconnect import KiteConnect
    except ImportError as exc:  # pragma: no cover - only on live path
        raise BrokerError("kiteconnect is not installed - `pip install kiteconnect` to trade live") from exc
    kite = KiteConnect(api_key=api_key)
    if access_token:
        kite.set_access_token(access_token)
    return ZerodhaAdapter(kite, api_key, api_secret, account_id)
