"""Unit tests for BrokerExecutionVenue against a fake BrokerAdapter. Proves
the same venue interface the OMS uses works for a broker, and that async order
updates flow to the event sink. Pure."""

from decimal import Decimal
from uuid import uuid4

from libs.broker.base import BrokerAdapter, BrokerError, BrokerSession
from libs.execution.broker_venue import BrokerExecutionVenue
from libs.trading_domain.entities import Margin, OrderEvent, OrderIntent, Position
from libs.trading_domain.enums import OrderState, OrderType, Product, Side

ACCOUNT = uuid4()


class FakeAdapter(BrokerAdapter):
    def __init__(self):
        self.placed = []
        self.cancelled = []
        self.fail = False

    def login_url(self):
        return "url"

    async def complete_login(self, callback):
        return BrokerSession("fake", "tok")

    def set_session(self, session):
        pass

    async def place_order(self, intent):
        if self.fail:
            raise BrokerError("rejected by RMS")
        self.placed.append(intent)
        return f"OID-{len(self.placed)}"

    async def modify_order(self, order_id, intent):
        return order_id

    async def cancel_order(self, order_id, variety="regular"):
        self.cancelled.append(order_id)
        return order_id

    async def get_positions(self):
        return [Position(ACCOUNT, "INFY", Product.MIS, 10, Decimal("1500"), Decimal("0"), Decimal("1510"))]

    async def get_holdings(self):
        return []

    async def available_margin(self):
        return Margin(available=Decimal("100000"))

    def parse_order_update(self, payload, order_to_intent):
        return OrderEvent(
            venue_order_id=payload["order_id"],
            intent_id=order_to_intent.get(payload["order_id"]),
            state=OrderState.COMPLETE if payload["status"] == "COMPLETE" else OrderState.OPEN,
            filled_qty=payload.get("filled_quantity", 0),
            pending_qty=payload.get("pending_quantity", 0),
            average_price=Decimal(str(payload.get("average_price", 0))),
        )


def _intent():
    return OrderIntent(uuid4(), ACCOUNT, "INFY", Side.BUY, OrderType.MARKET, Product.MIS, 10)


async def test_place_returns_submitted_and_tracks_order():
    venue = BrokerExecutionVenue(FakeAdapter())
    intent = _intent()
    ack = await venue.place(intent)
    assert ack.state is OrderState.SUBMITTED
    assert ack.venue_order_id == "OID-1"
    assert venue._order_to_intent["OID-1"] == intent.intent_id


async def test_place_failure_returns_rejected():
    adapter = FakeAdapter()
    adapter.fail = True
    venue = BrokerExecutionVenue(adapter)
    ack = await venue.place(_intent())
    assert ack.state is OrderState.REJECTED
    assert "RMS" in ack.reason


async def test_order_update_flows_to_event_sink_linked_to_intent():
    venue = BrokerExecutionVenue(FakeAdapter())
    events = []
    venue.set_event_sink(events.append)
    intent = _intent()
    ack = await venue.place(intent)
    venue.handle_order_update(
        {
            "order_id": ack.venue_order_id,
            "status": "COMPLETE",
            "filled_quantity": 10,
            "pending_quantity": 0,
            "average_price": 1500,
        }
    )
    assert len(events) == 1
    assert events[0].state is OrderState.COMPLETE
    assert events[0].intent_id == intent.intent_id


async def test_cancel_and_positions_and_margin_delegate():
    adapter = FakeAdapter()
    venue = BrokerExecutionVenue(adapter)
    await venue.place(_intent())
    cancel = await venue.cancel("OID-1")
    assert cancel.state is OrderState.CANCELLED
    assert adapter.cancelled == ["OID-1"]
    positions = await venue.positions()
    assert positions[0].symbol == "INFY"
    margin = await venue.available_margin()
    assert margin.available == Decimal("100000")
