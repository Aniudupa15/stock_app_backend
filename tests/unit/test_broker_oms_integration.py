"""The Phase 2 P1 payoff, proven: the SAME OrderManagementSystem that was
tested against the paper venue now drives a BrokerExecutionVenue. Broker fills
arrive asynchronously via the order-update stream (simulated here), and the
OMS still manages the bracket, runs OCO, and records the round-trip Trade.
Pure - no kiteconnect, no network."""

from decimal import Decimal
from uuid import uuid4

from libs.broker.base import BrokerAdapter, BrokerSession
from libs.charges.nse_equity import compute
from libs.execution.broker_venue import BrokerExecutionVenue
from libs.oms.oms import OrderManagementSystem
from libs.trading_domain.entities import BracketSpec, Margin, OrderEvent, OrderIntent, Position
from libs.trading_domain.enums import ExitReason, OrderState, OrderType, Product, Side

ACCOUNT = uuid4()


class FakeAdapter(BrokerAdapter):
    """Minimal broker: records placed orders and lets the test push updates."""

    def __init__(self):
        self.orders: dict[str, OrderIntent] = {}
        self.cancelled: list[str] = []
        self._seq = 0

    def login_url(self):
        return "url"

    async def complete_login(self, callback):
        return BrokerSession("fake", "tok")

    def set_session(self, session):
        pass

    async def place_order(self, intent):
        self._seq += 1
        oid = f"OID-{self._seq}"
        self.orders[oid] = intent
        return oid

    async def modify_order(self, order_id, intent):
        return order_id

    async def cancel_order(self, order_id, variety="regular"):
        self.cancelled.append(order_id)
        return order_id

    async def get_positions(self):
        return [Position(ACCOUNT, "INFY", Product.MIS, 0, Decimal("0"), Decimal("0"))]

    async def get_holdings(self):
        return []

    async def available_margin(self):
        return Margin(available=Decimal("1000000"))

    def parse_order_update(self, payload, order_to_intent):
        filled = payload.get("filled_quantity", 0)
        pending = payload.get("pending_quantity", 0)
        status = payload["status"]
        state = OrderState.COMPLETE if status == "COMPLETE" else OrderState.OPEN
        return OrderEvent(
            venue_order_id=payload["order_id"],
            intent_id=order_to_intent.get(payload["order_id"]),
            state=state,
            filled_qty=filled,
            pending_qty=pending,
            average_price=Decimal(str(payload.get("average_price", 0))),
        )


async def test_same_oms_drives_broker_venue_full_bracket_lifecycle():
    adapter = FakeAdapter()
    venue = BrokerExecutionVenue(adapter)
    oms = OrderManagementSystem(venue, clock=lambda: None)

    # 1. Submit a long bracket entry (MARKET) with SL 980 / target 1020.
    entry = OrderIntent(uuid4(), ACCOUNT, "INFY", Side.BUY, OrderType.MARKET, Product.MIS, 10)
    result = await oms.submit(entry, BracketSpec(stop_loss=Decimal("980"), target=Decimal("1020")))
    assert result.accepted
    assert oms.bracket_state(result.bracket_id) == "PENDING_ENTRY"
    entry_oid = result.ack.venue_order_id

    # 2. Broker confirms the entry fill @ 1000 -> OMS places SL + target children.
    venue.handle_order_update(
        {
            "order_id": entry_oid,
            "status": "COMPLETE",
            "filled_quantity": 10,
            "pending_quantity": 0,
            "average_price": 1000,
        }
    )
    await oms.process()
    assert oms.bracket_state(result.bracket_id) == "ACTIVE"
    assert len(adapter.orders) == 3  # entry + SL + target placed at the broker

    # 3. Broker confirms the target fill @ 1020 -> OCO cancels the SL, trade recorded.
    target_oid = [oid for oid, i in adapter.orders.items() if i.order_type is OrderType.LIMIT][0]
    venue.handle_order_update(
        {
            "order_id": target_oid,
            "status": "COMPLETE",
            "filled_quantity": 10,
            "pending_quantity": 0,
            "average_price": 1020,
        }
    )
    await oms.process()

    assert oms.bracket_state(result.bracket_id) == "CLOSED"
    sl_oid = [oid for oid, i in adapter.orders.items() if i.order_type is OrderType.SL_M][0]
    assert sl_oid in adapter.cancelled  # OCO cancelled the stop
    assert len(oms.trades) == 1
    trade = oms.trades[0]
    assert trade.exit_reason is ExitReason.TARGET
    assert trade.pnl_gross == Decimal("200")  # (1020-1000)*10
    # charges computed via libs/charges fallback (no paper fills on broker path)
    expected_charges = (
        compute(Side.BUY, Product.MIS, 10, Decimal("1000")).total
        + compute(Side.SELL, Product.MIS, 10, Decimal("1020")).total
    )
    assert trade.charges_total == expected_charges
    assert trade.pnl_net == Decimal("200") - expected_charges
