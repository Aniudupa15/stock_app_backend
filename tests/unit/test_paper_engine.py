"""End-to-end paper-trading loop test: tick -> white-box signal -> risk gate
-> OMS bracket -> managed exit -> recorded trade. Real venue + OMS + risk +
strategy, no mocks. Pure - no DB, no network."""

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from libs.engine.paper_engine import PaperTradingEngine
from libs.execution.paper import PaperExecutionVenue
from libs.execution.slippage import SlippageModel
from libs.oms.oms import OrderManagementSystem
from libs.risk.gate import RiskGate, RiskProfile
from libs.strategy.engine import ExitRule, Strategy
from libs.trading_calendar.calendar import TradingCalendar
from libs.trading_domain.entities import MarketQuote
from libs.trading_domain.enums import Product, Side

MON = datetime(2026, 1, 5, 10, 0)
NO_SLIP = SlippageModel(base_bps=Decimal("0"), impact_coeff_bps=Decimal("0"), illiquid_penalty_bps=Decimal("0"))
ACCOUNT = uuid4()

# close > EMA_20 -> go long
RULE = {"op": "GT", "left": {"feature": "close"}, "right": {"feature": "EMA_20"}}


def _quote(ltp):
    return MarketQuote(symbol="INFY", ltp=Decimal(ltp), day_volume=1_000_000)


def _engine(profile=None, quantity=10):
    quotes_holder = {"q": _quote("1000")}

    class Quotes:
        async def get_quote(self, symbol):
            return quotes_holder["q"]

    venue = PaperExecutionVenue(ACCOUNT, Quotes(), TradingCalendar(), lambda: MON, starting_cash=Decimal("10000000"), slippage=NO_SLIP)
    oms = OrderManagementSystem(venue, lambda: MON)
    strategy = Strategy(
        name="EMA breakout",
        rule=RULE,
        side=Side.BUY,
        product=Product.MIS,
        exit=ExitRule(stop_loss_pct=Decimal("2"), target_pct=Decimal("2")),
        quantity=quantity,
        strategy_id=uuid4(),
    )
    engine = PaperTradingEngine(
        ACCOUNT, venue, oms, RiskGate(), profile or RiskProfile(), TradingCalendar(), lambda: MON, [strategy]
    )
    return engine, venue, oms, quotes_holder


async def test_signal_fires_opens_bracketed_position():
    engine, venue, oms, _ = _engine()
    signal = await engine.process_tick("INFY", _quote("1000"), features={"close": 1010, "EMA_20": 1000})
    assert signal is not None
    positions = await venue.positions()
    assert positions[0].net_qty == 10
    assert positions[0].avg_price == Decimal("1000")


async def test_no_signal_no_position():
    engine, venue, _, _ = _engine()
    signal = await engine.process_tick("INFY", _quote("1000"), features={"close": 990, "EMA_20": 1000})
    assert signal is None
    assert await venue.positions() == []


async def test_full_round_trip_target_hit_records_trade():
    engine, venue, oms, quotes = _engine()
    # Entry tick
    await engine.process_tick("INFY", _quote("1000"), features={"close": 1010, "EMA_20": 1000})
    # Price rallies past +2% target (1020)
    quotes["q"] = _quote("1021")
    await engine.process_tick("INFY", _quote("1021"))
    assert len(oms.trades) == 1
    trade = oms.trades[0]
    assert trade.exit_price == Decimal("1020.00")
    assert trade.pnl_gross == Decimal("200")  # (1020-1000)*10
    assert (await venue.positions())[0].net_qty == 0  # flat again


async def test_no_duplicate_entry_while_position_open():
    engine, venue, oms, _ = _engine()
    await engine.process_tick("INFY", _quote("1000"), features={"close": 1010, "EMA_20": 1000})
    # Same firing features again -> must NOT open a second position.
    second = await engine.process_tick("INFY", _quote("1000"), features={"close": 1010, "EMA_20": 1000})
    assert second is None
    assert (await venue.positions())[0].net_qty == 10


async def test_risk_gate_resizes_entry_down():
    # per-trade risk 1% of 10,000,000 = 100,000; SL distance = 1000-980 = 20 -> max 5000 qty.
    # Request 100,000 -> resized to 5000.
    profile = RiskProfile(per_trade_risk_pct=Decimal("1"))
    engine, venue, oms, _ = _engine(profile=profile, quantity=100000)
    await engine.process_tick("INFY", _quote("1000"), features={"close": 1010, "EMA_20": 1000})
    positions = await venue.positions()
    assert positions[0].net_qty == 5000
