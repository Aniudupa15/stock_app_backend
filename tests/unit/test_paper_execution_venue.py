"""Unit tests for the PaperExecutionVenue. Pure - no DB, no network, no Docker.

Slippage is set to zero in most tests so fill prices (and therefore P&L and
cash) are exact and hand-checkable; one test exercises slippage explicitly.
"""

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from libs.charges.nse_equity import compute
from libs.execution.paper import PaperExecutionVenue
from libs.execution.slippage import SlippageModel
from libs.trading_calendar.calendar import TradingCalendar
from libs.trading_domain.entities import MarketQuote, OrderIntent
from libs.trading_domain.enums import OrderState, OrderType, Product, Side

MON = datetime(2026, 1, 5, 10, 0)  # Monday, continuous session
NO_SLIP = SlippageModel(base_bps=Decimal("0"), impact_coeff_bps=Decimal("0"), illiquid_penalty_bps=Decimal("0"))
ACCOUNT = uuid4()


class FakeQuotes:
    def __init__(self, quote: MarketQuote):
        self.quote = quote

    def set(self, quote: MarketQuote):
        self.quote = quote

    async def get_quote(self, symbol: str) -> MarketQuote:
        return self.quote


def _intent(**kw):
    base = dict(
        intent_id=uuid4(),
        account_id=ACCOUNT,
        symbol="INFY",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        product=Product.MIS,
        quantity=100,
    )
    base.update(kw)
    return OrderIntent(**base)


def _q(ltp="1000", **kw):
    return MarketQuote(symbol="INFY", ltp=Decimal(ltp), day_volume=kw.pop("day_volume", 1_000_000), **kw)


async def test_market_buy_fills_and_debits_cash():
    quote = _q("1000")
    venue = PaperExecutionVenue(
        ACCOUNT, FakeQuotes(quote), TradingCalendar(), lambda: MON, starting_cash=Decimal("100000"), slippage=NO_SLIP
    )
    ack = await venue.place(_intent(side=Side.BUY, quantity=10, product=Product.MIS))
    assert ack.state is OrderState.COMPLETE
    charges = compute(Side.BUY, Product.MIS, 10, Decimal("1000"))
    # cash = 100000 - (10*1000) - charges
    assert venue.cash() == Decimal("100000") - Decimal("10000") - charges.total
    positions = await venue.positions()
    assert len(positions) == 1
    assert positions[0].net_qty == 10
    assert positions[0].avg_price == Decimal("1000")


async def test_market_closed_rejects():
    quote = _q("1000")
    closed = datetime(2026, 1, 5, 8, 0)  # before pre-open
    venue = PaperExecutionVenue(
        ACCOUNT, FakeQuotes(quote), TradingCalendar(), lambda: closed, starting_cash=Decimal("100000")
    )
    ack = await venue.place(_intent())
    assert ack.state is OrderState.REJECTED
    assert ack.reason == "market closed"


async def test_circuit_locked_rejects_buy_at_upper_band():
    quote = _q("1050", upper_band=Decimal("1050"))
    venue = PaperExecutionVenue(
        ACCOUNT, FakeQuotes(quote), TradingCalendar(), lambda: MON, starting_cash=Decimal("100000")
    )
    ack = await venue.place(_intent(side=Side.BUY))
    assert ack.state is OrderState.REJECTED
    assert ack.reason == "circuit locked"


async def test_round_trip_realized_pnl_and_cash():
    # Buy 100 @ 1000, then sell 100 @ 1010 (delivery). Realized = (1010-1000)*100 = 1000 gross.
    quotes = FakeQuotes(_q("1000"))
    venue = PaperExecutionVenue(
        ACCOUNT, quotes, TradingCalendar(), lambda: MON, starting_cash=Decimal("1000000"), slippage=NO_SLIP
    )
    await venue.place(_intent(side=Side.BUY, quantity=100, product=Product.CNC))
    quotes.set(_q("1010"))
    await venue.place(_intent(side=Side.SELL, quantity=100, product=Product.CNC))
    positions = await venue.positions()
    assert positions[0].net_qty == 0
    assert positions[0].realized_pnl == Decimal("1000")  # gross of charges (charges hit cash separately)
    buy_ch = compute(Side.BUY, Product.CNC, 100, Decimal("1000"))
    sell_ch = compute(Side.SELL, Product.CNC, 100, Decimal("1010"))
    expected_cash = Decimal("1000000") - Decimal("100000") - buy_ch.total + Decimal("101000") - sell_ch.total
    assert venue.cash() == expected_cash


async def test_resting_limit_buy_fills_on_tick_when_price_drops():
    quotes = FakeQuotes(_q("1000"))
    venue = PaperExecutionVenue(
        ACCOUNT, quotes, TradingCalendar(), lambda: MON, starting_cash=Decimal("100000"), slippage=NO_SLIP
    )
    ack = await venue.place(_intent(side=Side.BUY, order_type=OrderType.LIMIT, price=Decimal("990"), quantity=10))
    assert ack.state is OrderState.OPEN  # not marketable at 1000
    assert (await venue.positions()) == []
    await venue.on_tick(_q("989"))  # crosses the limit
    positions = await venue.positions()
    assert positions[0].net_qty == 10
    assert positions[0].avg_price == Decimal("990")  # filled at the limit price


async def test_resting_stop_loss_sell_triggers_on_drop():
    quotes = FakeQuotes(_q("1000"))
    venue = PaperExecutionVenue(
        ACCOUNT, quotes, TradingCalendar(), lambda: MON, starting_cash=Decimal("100000"), slippage=NO_SLIP
    )
    # Long already established
    await venue.place(_intent(side=Side.BUY, quantity=10, product=Product.MIS))
    # SL-M sell with trigger 980
    ack = await venue.place(
        _intent(
            side=Side.SELL, order_type=OrderType.SL_M, trigger_price=Decimal("980"), quantity=10, product=Product.MIS
        )
    )
    assert ack.state is OrderState.OPEN
    await venue.on_tick(_q("979"))  # breaches stop
    positions = await venue.positions()
    assert positions[0].net_qty == 0  # squared off


async def test_cancel_resting_order():
    quotes = FakeQuotes(_q("1000"))
    venue = PaperExecutionVenue(ACCOUNT, quotes, TradingCalendar(), lambda: MON, starting_cash=Decimal("100000"))
    ack = await venue.place(_intent(side=Side.BUY, order_type=OrderType.LIMIT, price=Decimal("990")))
    cancel = await venue.cancel(ack.venue_order_id)
    assert cancel.state is OrderState.CANCELLED
    await venue.on_tick(_q("980"))  # would have filled if not cancelled
    assert (await venue.positions()) == []


async def test_event_sink_receives_fill_events():
    quotes = FakeQuotes(_q("1000"))
    venue = PaperExecutionVenue(ACCOUNT, quotes, TradingCalendar(), lambda: MON, starting_cash=Decimal("100000"))
    events = []
    venue.set_event_sink(events.append)
    await venue.place(_intent(side=Side.BUY, quantity=5))
    assert any(e.state is OrderState.COMPLETE and e.filled_qty == 5 for e in events)


async def test_slippage_makes_buy_fill_higher():
    quotes = FakeQuotes(_q("1000", day_volume=1_000_000))
    slip = SlippageModel(base_bps=Decimal("10"), impact_coeff_bps=Decimal("0"), illiquid_penalty_bps=Decimal("0"))
    venue = PaperExecutionVenue(
        ACCOUNT, quotes, TradingCalendar(), lambda: MON, starting_cash=Decimal("10000000"), slippage=slip
    )
    await venue.place(_intent(side=Side.BUY, quantity=10))
    # 10 bps above 1000 = 1001
    positions = await venue.positions()
    assert positions[0].avg_price == Decimal("1001.000")


async def test_available_margin_reflects_cash():
    quotes = FakeQuotes(_q("1000"))
    venue = PaperExecutionVenue(ACCOUNT, quotes, TradingCalendar(), lambda: MON, starting_cash=Decimal("55000"))
    margin = await venue.available_margin()
    assert margin.available == Decimal("55000")
