"""Unit tests for the Zerodha adapter mapping + a fake Kite client. Pure -
no kiteconnect package, no credentials, no network."""

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from libs.broker.base import BrokerError
from libs.broker.zerodha import ZerodhaAdapter, intent_to_kite_params, kite_status_to_state
from libs.trading_domain.entities import OrderIntent
from libs.trading_domain.enums import OrderState, OrderType, Product, Side

ACCOUNT = uuid4()


class FakeKite:
    def __init__(self):
        self.placed = []
        self.access_token = None
        self.raise_on_place = False

    def generate_session(self, request_token, api_secret):
        return {"access_token": f"tok-{request_token}"}

    def set_access_token(self, token):
        self.access_token = token

    def place_order(self, **params):
        if self.raise_on_place:
            raise RuntimeError("insufficient funds")
        self.placed.append(params)
        return "2500001"

    def cancel_order(self, variety, order_id):
        return order_id

    def positions(self):
        return {
            "net": [
                {"tradingsymbol": "INFY", "product": "MIS", "quantity": 10, "average_price": 1500.5, "realised": 0, "last_price": 1512.0}
            ]
        }

    def holdings(self):
        return [{"tradingsymbol": "TCS", "quantity": 5, "average_price": 3400.0, "last_price": 3450.0}]

    def margins(self, segment):
        return {"available": {"live_balance": 250000.0, "cash": 250000.0}}


def _intent(**kw):
    base = dict(
        intent_id=uuid4(),
        account_id=ACCOUNT,
        symbol="INFY",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        product=Product.MIS,
        quantity=10,
    )
    base.update(kw)
    return OrderIntent(**base)


def _adapter(kite=None):
    return ZerodhaAdapter(kite or FakeKite(), "apikey", "apisecret", ACCOUNT, clock=lambda: datetime(2026, 1, 5, 10, 0))


def test_market_order_params():
    p = intent_to_kite_params(_intent(order_type=OrderType.MARKET))
    assert p["variety"] == "regular"
    assert p["transaction_type"] == "BUY"
    assert p["order_type"] == "MARKET"
    assert p["product"] == "MIS"
    assert "price" not in p and "trigger_price" not in p
    assert len(p["tag"]) <= 20


def test_sl_m_order_maps_hyphenated_and_carries_trigger():
    p = intent_to_kite_params(_intent(side=Side.SELL, order_type=OrderType.SL_M, trigger_price=Decimal("980")))
    assert p["order_type"] == "SL-M"
    assert p["trigger_price"] == 980.0
    assert "price" not in p


def test_limit_order_carries_price():
    p = intent_to_kite_params(_intent(order_type=OrderType.LIMIT, price=Decimal("1490")))
    assert p["order_type"] == "LIMIT"
    assert p["price"] == 1490.0


def test_status_mapping_partial_and_complete():
    assert kite_status_to_state("COMPLETE", 10, 0) is OrderState.COMPLETE
    assert kite_status_to_state("REJECTED", 0, 10) is OrderState.REJECTED
    assert kite_status_to_state("OPEN", 4, 6) is OrderState.PARTIAL  # partial fill
    assert kite_status_to_state("OPEN", 0, 10) is OrderState.OPEN
    assert kite_status_to_state("TRIGGER PENDING", 0, 10) is OrderState.OPEN
    assert kite_status_to_state("PUT ORDER REQ RECEIVED", 0, 10) is OrderState.SUBMITTED


async def test_place_order_returns_broker_id_and_sends_params():
    kite = FakeKite()
    adapter = _adapter(kite)
    order_id = await adapter.place_order(_intent())
    assert order_id == "2500001"
    assert kite.placed[0]["tradingsymbol"] == "INFY"


async def test_place_order_wraps_failure_in_broker_error():
    kite = FakeKite()
    kite.raise_on_place = True
    adapter = _adapter(kite)
    with pytest.raises(BrokerError):
        await adapter.place_order(_intent())


async def test_complete_login_sets_token_and_expiry():
    kite = FakeKite()
    adapter = _adapter(kite)
    session = await adapter.complete_login({"request_token": "rt123"})
    assert session.access_token == "tok-rt123"
    assert kite.access_token == "tok-rt123"
    # clock is 2026-01-05 10:00 -> expiry is next day 06:00
    assert session.expires_at.date() == datetime(2026, 1, 6).date()
    assert session.expires_at.hour == 6


async def test_get_positions_maps_net():
    positions = await _adapter().get_positions()
    assert positions[0].symbol == "INFY"
    assert positions[0].net_qty == 10
    assert positions[0].avg_price == Decimal("1500.5")
    assert positions[0].ltp == Decimal("1512.0")


async def test_available_margin_reads_live_balance():
    margin = await _adapter().available_margin()
    assert margin.available == Decimal("250000.0")


def test_parse_order_update_builds_event_linked_to_intent():
    adapter = _adapter()
    intent_id = uuid4()
    payload = {"order_id": "2500001", "status": "COMPLETE", "filled_quantity": 10, "pending_quantity": 0, "average_price": 1501.25}
    event = adapter.parse_order_update(payload, {"2500001": intent_id})
    assert event.state is OrderState.COMPLETE
    assert event.intent_id == intent_id
    assert event.filled_qty == 10
    assert event.average_price == Decimal("1501.25")


def test_login_url_contains_api_key():
    assert "api_key=apikey" in _adapter().login_url()
