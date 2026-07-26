"""Unit tests for the OMS: managed bracket, OCO, trailing SL, round-trip Trade.

Runs against the real PaperExecutionVenue (no mocks) with zero slippage so
prices/P&L are exact. Pure - no DB, no network.
"""

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from libs.charges.nse_equity import compute
from libs.execution.paper import PaperExecutionVenue
from libs.execution.slippage import SlippageModel
from libs.oms.oms import OrderManagementSystem
from libs.trading_calendar.calendar import TradingCalendar
from libs.trading_domain.entities import BracketSpec, MarketQuote, OrderIntent, TrailingSpec
from libs.trading_domain.enums import ExitReason, OrderType, Product, Side

MON = datetime(2026, 1, 5, 10, 0)
NO_SLIP = SlippageModel(base_bps=Decimal("0"), impact_coeff_bps=Decimal("0"), illiquid_penalty_bps=Decimal("0"))
ACCOUNT = uuid4()


class FakeQuotes:
    def __init__(self, quote):
        self.quote = quote

    def set(self, quote):
        self.quote = quote

    async def get_quote(self, symbol):
        return self.quote


def _q(ltp):
    return MarketQuote(symbol="INFY", ltp=Decimal(ltp), day_volume=1_000_000)


def _setup(ltp="1000"):
    quotes = FakeQuotes(_q(ltp))
    venue = PaperExecutionVenue(ACCOUNT, quotes, TradingCalendar(), lambda: MON, starting_cash=Decimal("10000000"), slippage=NO_SLIP)
    oms = OrderManagementSystem(venue, lambda: MON)
    return quotes, venue, oms


def _entry(side=Side.BUY, qty=10):
    return OrderIntent(uuid4(), ACCOUNT, "INFY", side, OrderType.MARKET, Product.MIS, qty)


async def test_bracket_children_placed_after_entry_fills():
    _, venue, oms = _setup("1000")
    res = await oms.submit(_entry(), BracketSpec(stop_loss=Decimal("980"), target=Decimal("1020")))
    assert res.accepted
    assert oms.bracket_state(res.bracket_id) == "PENDING_ENTRY"
    await oms.process()  # entry filled -> place SL + target
    assert oms.bracket_state(res.bracket_id) == "ACTIVE"
    # entry + SL + target = 3 orders placed; SL/target resting
    positions = await venue.positions()
    assert positions[0].net_qty == 10


async def test_target_hit_records_trade_and_cancels_stop():
    quotes, venue, oms = _setup("1000")
    res = await oms.submit(_entry(), BracketSpec(stop_loss=Decimal("980"), target=Decimal("1020")))
    await oms.process()
    quotes.set(_q("1021"))
    await venue.on_tick(_q("1021"))  # target LIMIT sell @1020 fills
    await oms.process()  # OCO -> cancel SL, close bracket, record trade
    assert oms.bracket_state(res.bracket_id) == "CLOSED"
    assert len(oms.trades) == 1
    trade = oms.trades[0]
    assert trade.exit_reason is ExitReason.TARGET
    assert trade.entry_price == Decimal("1000")
    assert trade.exit_price == Decimal("1020")
    assert trade.pnl_gross == Decimal("200")  # (1020-1000)*10
    expected_charges = compute(Side.BUY, Product.MIS, 10, Decimal("1000")).total + compute(Side.SELL, Product.MIS, 10, Decimal("1020")).total
    assert trade.charges_total == expected_charges
    assert trade.pnl_net == Decimal("200") - expected_charges
    # position flat after exit
    assert (await venue.positions())[0].net_qty == 0


async def test_stop_loss_hit_records_losing_trade():
    quotes, venue, oms = _setup("1000")
    res = await oms.submit(_entry(), BracketSpec(stop_loss=Decimal("980"), target=Decimal("1020")))
    await oms.process()
    quotes.set(_q("979"))
    await venue.on_tick(_q("979"))  # SL-M sell trigger 980 breached
    await oms.process()
    assert oms.bracket_state(res.bracket_id) == "CLOSED"
    trade = oms.trades[0]
    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert trade.pnl_gross == Decimal("-210")  # (979-1000)*10


async def test_rejected_entry_returns_not_accepted():
    # Market closed -> venue rejects entry -> OMS reports not accepted.
    quotes = FakeQuotes(_q("1000"))
    closed = datetime(2026, 1, 5, 8, 0)
    venue = PaperExecutionVenue(ACCOUNT, quotes, TradingCalendar(), lambda: closed, starting_cash=Decimal("1000000"))
    oms = OrderManagementSystem(venue, lambda: closed)
    res = await oms.submit(_entry(), BracketSpec(stop_loss=Decimal("980")))
    assert res.accepted is False
    assert res.reason == "market closed"


async def test_trailing_stop_ratchets_up_on_favorable_move():
    _, venue, oms = _setup("1000")
    res = await oms.submit(
        _entry(), BracketSpec(stop_loss=Decimal("980"), target=Decimal("1100"), trailing=TrailingSpec(by="PCT", value=Decimal("2")))
    )
    await oms.process()
    bracket = oms._brackets[res.bracket_id]
    assert bracket.current_sl == Decimal("980")
    # price rises to 1010 -> new SL = 1010 - 2% = 989.8 (raises)
    await oms.on_tick(_q("1010"))
    assert bracket.current_sl == Decimal("989.8")
    # price dips to 1005 -> trailing never lowers the stop
    await oms.on_tick(_q("1005"))
    assert bracket.current_sl == Decimal("989.8")
