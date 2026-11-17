"""F&O paper trading: charges, contracts, margin, venue accounting, risk
sizing, OMS square-off/expiry, and a full simulated intraday session through
the Zerodha read-only wrapper. Pure - no network, no kiteconnect, no DB.

Prices in the session test are a scripted synthetic path (not market data);
it exercises the mechanics, not a strategy's edge."""

from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

from libs.broker.kite_quotes import KiteQuoteSource, parse_kite_quote
from libs.broker.zerodha import ZerodhaAdapter, intent_to_kite_params
from libs.charges import nse_equity, nse_fno
from libs.charges.router import charges_for
from libs.derivatives.contracts import (
    ContractRegistry,
    FnoContract,
    InstrumentKind,
    contracts_from_kite_instruments,
    kind_from_fno_symbol,
)
from libs.derivatives.margin import FnoMarginModel
from libs.engine.paper_engine import PaperTradingEngine
from libs.execution.paper import PaperExecutionVenue
from libs.execution.slippage import SlippageModel
from libs.oms.oms import OrderManagementSystem
from libs.risk.gate import RiskDecision, RiskGate, RiskProfile, RiskState
from libs.strategy.engine import ExitRule, Strategy
from libs.trading_calendar.calendar import IST, TradingCalendar
from libs.trading_domain.entities import BracketSpec, MarketQuote, OrderIntent
from libs.trading_domain.enums import ExitReason, OrderState, OrderType, Product, Side
from services.trading_service.fno_paper import (
    ReadOnlyKite,
    SessionConfig,
    StreamingIndicators,
    run_session,
)

MON = datetime(2026, 1, 5, 10, 0)
EXPIRY = date(2026, 1, 27)
NO_SLIP = SlippageModel(base_bps=Decimal("0"), impact_coeff_bps=Decimal("0"), illiquid_penalty_bps=Decimal("0"))
ACCOUNT = uuid4()

FUT = FnoContract("NIFTY26JANFUT", "NIFTY", InstrumentKind.FUT, EXPIRY, 75)
CE = FnoContract("NIFTY26JAN24000CE", "NIFTY", InstrumentKind.CE, EXPIRY, 75, strike=Decimal("24000"))
PE = FnoContract("NIFTY26JAN24000PE", "NIFTY", InstrumentKind.PE, EXPIRY, 75, strike=Decimal("24000"))
STOCK_FUT = FnoContract("RELIANCE26JANFUT", "RELIANCE", InstrumentKind.FUT, EXPIRY, 500)


class Quotes:
    def __init__(self):
        self.prices: dict[str, Decimal] = {}

    def set(self, symbol, price):
        self.prices[symbol] = Decimal(str(price))

    async def get_quote(self, symbol):
        return MarketQuote(symbol=symbol, ltp=self.prices[symbol], day_volume=10_000_000)


def _venue(cash="500000", clock=lambda: MON):
    quotes = Quotes()
    registry = ContractRegistry([FUT, CE, PE, STOCK_FUT])
    venue = PaperExecutionVenue(
        ACCOUNT,
        quotes,
        TradingCalendar(),
        clock,
        starting_cash=Decimal(cash),
        slippage=NO_SLIP,
        contracts=registry,
    )
    return venue, quotes


def _intent(symbol, side, qty, **kw):
    base = dict(
        intent_id=uuid4(),
        account_id=ACCOUNT,
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        product=Product.NRML,
        quantity=qty,
        exchange="NFO",
    )
    base.update(kw)
    return OrderIntent(**base)


# ---------------------------------------------------------------- charges


def test_futures_charges_match_hand_calculation():
    buy = nse_fno.compute(Side.BUY, InstrumentKind.FUT, 75, Decimal("24000"))
    # turnover 18,00,000: brokerage capped 20, txn 31.14, sebi 1.80, stamp 36, gst 18% of 52.94
    assert (buy.brokerage, buy.stt, buy.exchange_txn, buy.sebi, buy.stamp_duty, buy.gst) == (
        Decimal("20.00"),
        Decimal("0.00"),
        Decimal("31.14"),
        Decimal("1.80"),
        Decimal("36.00"),
        Decimal("9.53"),
    )
    assert buy.total == Decimal("98.47")
    sell = nse_fno.compute(Side.SELL, InstrumentKind.FUT, 75, Decimal("24100"))
    assert sell.stt == Decimal("903.75")  # 0.05% of 18,07,500
    assert sell.stamp_duty == Decimal("0.00")
    assert sell.total == Decimal("966.38")


def test_option_charges_flat_brokerage_on_premium():
    buy = nse_fno.compute(Side.BUY, InstrumentKind.CE, 75, Decimal("100"))
    assert buy.brokerage == Decimal("20.00")
    assert buy.exchange_txn == Decimal("2.63")  # 0.03503% of 7,500 premium
    assert buy.stamp_duty == Decimal("0.23")
    assert buy.stt == Decimal("0.00")
    sell = nse_fno.compute(Side.SELL, InstrumentKind.PE, 75, Decimal("100"))
    assert sell.stt == Decimal("11.25")  # 0.15% of premium
    tiny = nse_fno.compute(Side.SELL, InstrumentKind.CE, 75, Decimal("0.10"))
    assert tiny.brokerage == Decimal("7.50")  # never more than the premium


def test_settlement_charges_only_on_long_itm_options():
    assert nse_fno.compute_settlement(InstrumentKind.CE, 75, Decimal("150")).stt == Decimal("16.88")
    assert nse_fno.compute_settlement(InstrumentKind.CE, -75, Decimal("150")).total == Decimal("0")
    assert nse_fno.compute_settlement(InstrumentKind.FUT, 75, Decimal("24000")).total == Decimal("0")


def test_router_uses_exchange_not_symbol_suffix():
    # "ACE" is a real NSE equity ending in "CE" - must still get equity charges.
    eq = charges_for("NSE", "ACE", Side.BUY, Product.MIS, 10, Decimal("1000"))
    assert eq == nse_equity.compute(Side.BUY, Product.MIS, 10, Decimal("1000"))
    fo = charges_for("NFO", "NIFTY26JAN24000CE", Side.BUY, Product.NRML, 75, Decimal("100"))
    assert fo == nse_fno.compute(Side.BUY, InstrumentKind.CE, 75, Decimal("100"))
    with pytest.raises(ValueError):
        nse_equity.compute(Side.BUY, Product.NRML, 1, Decimal("1"))


# ---------------------------------------------------------------- contracts


KITE_ROWS = [
    {"instrument_token": 1, "tradingsymbol": "NIFTY26JANFUT", "name": "NIFTY", "expiry": date(2026, 1, 27),
     "strike": 0.0, "tick_size": 0.1, "lot_size": 75, "instrument_type": "FUT", "exchange": "NFO"},
    {"instrument_token": 2, "tradingsymbol": "NIFTY26FEBFUT", "name": "NIFTY", "expiry": date(2026, 2, 24),
     "strike": 0.0, "tick_size": 0.1, "lot_size": 75, "instrument_type": "FUT", "exchange": "NFO"},
    {"instrument_token": 3, "tradingsymbol": "NIFTY2610624000CE", "name": "NIFTY", "expiry": date(2026, 1, 6),
     "strike": 24000.0, "tick_size": 0.05, "lot_size": 75, "instrument_type": "CE", "exchange": "NFO"},
    {"instrument_token": 4, "tradingsymbol": "NIFTY2610624050CE", "name": "NIFTY", "expiry": date(2026, 1, 6),
     "strike": 24050.0, "tick_size": 0.05, "lot_size": 75, "instrument_type": "CE", "exchange": "NFO"},
    {"instrument_token": 5, "tradingsymbol": "NIFTY26JAN24000CE", "name": "NIFTY", "expiry": date(2026, 1, 27),
     "strike": 24000.0, "tick_size": 0.05, "lot_size": 75, "instrument_type": "CE", "exchange": "NFO"},
    {"instrument_token": 6, "tradingsymbol": "NIFTY", "name": "NIFTY", "expiry": "", "strike": 0,
     "tick_size": 0.05, "lot_size": 0, "instrument_type": "EQ", "exchange": "NFO"},
]  # fmt: skip


def test_kite_instrument_parsing_and_selection():
    registry = ContractRegistry(contracts_from_kite_instruments(KITE_ROWS))
    assert len(registry) == 5  # EQ row skipped
    assert registry.nearest_future("NIFTY", today=date(2026, 1, 5)).tradingsymbol == "NIFTY26JANFUT"
    assert registry.nearest_future("NIFTY", today=date(2026, 1, 28)).tradingsymbol == "NIFTY26FEBFUT"
    atm = registry.atm_option("NIFTY", InstrumentKind.CE, Decimal("24030"), today=date(2026, 1, 5))
    assert atm.tradingsymbol == "NIFTY2610624050CE"  # nearest (weekly) expiry, closest strike
    assert atm.strike == Decimal("24050")
    assert registry.get("NIFTY26JANFUT").is_index


def test_settlement_value_and_symbol_kind():
    assert CE.settlement_price(Decimal("24150")) == Decimal("150")
    assert PE.settlement_price(Decimal("24150")) == Decimal("0")
    assert PE.settlement_price(Decimal("23900")) == Decimal("100")
    assert FUT.settlement_price(Decimal("24150")) == Decimal("24150")
    assert kind_from_fno_symbol("NIFTY26JAN24000PE") is InstrumentKind.PE
    with pytest.raises(ValueError):
        kind_from_fno_symbol("INFY")


def test_margin_model():
    m = FnoMarginModel()
    assert m.blocked(FUT, 75, Decimal("24000")) == Decimal("216000.00")  # 12% index
    assert m.blocked(STOCK_FUT, -500, Decimal("1300")) == Decimal("130000.00")  # 20% stock
    assert m.blocked(CE, 75, Decimal("100")) == 0  # long option: premium only
    assert m.blocked(CE, -75, Decimal("100")) == Decimal("216000.00")
    assert m.entry_cost_per_unit(CE, 1, Decimal("100")) == Decimal("100")


# ---------------------------------------------------------------- paper venue


async def test_futures_round_trip_is_margined_not_paid_for():
    venue, quotes = _venue()
    quotes.set(FUT.tradingsymbol, "24000")
    ack = await venue.place(_intent(FUT.tradingsymbol, Side.BUY, 75))
    assert ack.state is OrderState.COMPLETE
    assert venue.cash() == Decimal("500000") - Decimal("98.47")  # no 18-lakh notional debit
    assert venue.blocked_margin() == Decimal("216000.00")
    assert (await venue.available_margin()).available == venue.cash() - Decimal("216000.00")

    await venue.on_tick(MarketQuote(FUT.tradingsymbol, Decimal("24100")))
    assert venue.equity() == venue.cash() + Decimal("7500")  # unrealised MTM

    quotes.set(FUT.tradingsymbol, "24100")
    await venue.place(_intent(FUT.tradingsymbol, Side.SELL, 75))
    assert venue.cash() == Decimal("500000") - Decimal("98.47") + Decimal("7500") - Decimal("966.38")
    assert venue.blocked_margin() == 0
    assert venue.equity() == venue.cash()
    (pos,) = await venue.positions()
    assert pos.product is Product.NRML and pos.net_qty == 0 and pos.realized_pnl == Decimal("7500")


@pytest.mark.parametrize(
    ("intent_kw", "reason"),
    [
        ({"quantity": 50}, "multiple of lot size 75"),
        ({"product": Product.CNC}, "CNC not allowed"),
        ({"symbol": "NIFTY26DEC24000CE"}, "unknown F&O contract"),
    ],
)
async def test_fno_order_validation(intent_kw, reason):
    venue, quotes = _venue()
    quotes.set(intent_kw.get("symbol", FUT.tradingsymbol), "24000")
    kw = {"symbol": FUT.tradingsymbol, "side": Side.BUY, "qty": 75}
    kw["qty"] = intent_kw.pop("quantity", 75)
    symbol = intent_kw.pop("symbol", FUT.tradingsymbol)
    ack = await venue.place(_intent(symbol, Side.BUY, kw["qty"], **intent_kw))
    assert ack.state is OrderState.REJECTED
    assert reason in ack.reason


async def test_expired_contract_rejected():
    venue, quotes = _venue(clock=lambda: datetime(2026, 1, 28, 10, 0))
    quotes.set(FUT.tradingsymbol, "24000")
    ack = await venue.place(_intent(FUT.tradingsymbol, Side.BUY, 75))
    assert ack.reason == "contract expired"


async def test_insufficient_margin_rejects_entry_but_never_exit():
    venue, quotes = _venue(cash="300000")
    quotes.set(FUT.tradingsymbol, "24000")
    assert (await venue.place(_intent(FUT.tradingsymbol, Side.BUY, 75))).state is OrderState.COMPLETE
    second = await venue.place(_intent(FUT.tradingsymbol, Side.BUY, 75))  # would need another 2.16L
    assert second.state is OrderState.REJECTED and "insufficient margin" in second.reason
    exit_ack = await venue.place(_intent(FUT.tradingsymbol, Side.SELL, 75))
    assert exit_ack.state is OrderState.COMPLETE


async def test_option_buy_pays_premium_and_settles_itm_at_expiry():
    venue, quotes = _venue()
    quotes.set(CE.tradingsymbol, "100")
    await venue.place(_intent(CE.tradingsymbol, Side.BUY, 75))
    buy_charges = nse_fno.compute(Side.BUY, InstrumentKind.CE, 75, Decimal("100")).total
    assert venue.cash() == Decimal("500000") - Decimal("7500") - buy_charges
    assert venue.blocked_margin() == 0

    fills = venue.settle_expired(EXPIRY, {"NIFTY": Decimal("24150")})
    (fill,) = fills
    assert (fill.side, fill.qty, fill.price) == (Side.SELL, 75, Decimal("150"))
    assert venue.cash() == Decimal("500000") - Decimal("7500") - buy_charges + Decimal("11250") - Decimal("16.88")
    assert venue.settle_expired(EXPIRY, {"NIFTY": Decimal("24150")}) == []  # idempotent


async def test_short_option_blocks_margin_and_expires_worthless():
    venue, quotes = _venue()
    quotes.set(PE.tradingsymbol, "80")
    await venue.place(_intent(PE.tradingsymbol, Side.SELL, 75))
    sell_charges = nse_fno.compute(Side.SELL, InstrumentKind.PE, 75, Decimal("80")).total
    assert venue.cash() == Decimal("500000") + Decimal("6000") - sell_charges
    assert venue.blocked_margin() == Decimal("216000.00")
    # Not yet expired: nothing settles the day before.
    assert venue.settle_expired(EXPIRY - timedelta(days=1), {"NIFTY": Decimal("24150")}) == []
    venue.settle_expired(EXPIRY, {"NIFTY": Decimal("24150")})
    assert venue.blocked_margin() == 0
    assert venue.cash() == Decimal("500000") + Decimal("6000") - sell_charges  # kept the full premium


async def test_equity_path_unchanged_with_registry():
    venue, quotes = _venue()
    quotes.set("INFY", "1000")
    await venue.place(_intent("INFY", Side.BUY, 10, product=Product.MIS, exchange="NSE"))
    charges = nse_equity.compute(Side.BUY, Product.MIS, 10, Decimal("1000")).total
    assert venue.cash() == Decimal("500000") - Decimal("10000") - charges


# ---------------------------------------------------------------- risk gate


def test_risk_gate_rounds_to_whole_lots_and_sizes_on_margin():
    gate = RiskGate()
    intent = _intent(FUT.tradingsymbol, Side.BUY, 300)  # 4 lots requested
    state = RiskState(equity=Decimal("500000"), available_cash=Decimal("500000"))
    verdict = gate.evaluate(
        intent,
        state=state,
        profile=RiskProfile(),
        now=MON,
        calendar=TradingCalendar(),
        entry_price=Decimal("24000"),
        lot_size=75,
        funds_per_unit=Decimal("2880"),  # 12% of 24000
    )
    # 5,00,000 / 2,880 = 173 units -> 2 whole lots
    assert verdict.decision is RiskDecision.RESIZE and verdict.quantity == 150

    poor = RiskState(equity=Decimal("100000"), available_cash=Decimal("100000"))
    rejected = gate.evaluate(
        intent,
        state=poor,
        profile=RiskProfile(),
        now=MON,
        calendar=TradingCalendar(),
        entry_price=Decimal("24000"),
        lot_size=75,
        funds_per_unit=Decimal("2880"),
    )
    assert rejected.decision is RiskDecision.REJECT and "less than one lot" in rejected.reason


# ---------------------------------------------------------------- OMS + engine


async def test_oms_square_off_journals_trade_and_cancels_children():
    venue, quotes = _venue()
    oms = OrderManagementSystem(venue, lambda: MON)
    quotes.set(FUT.tradingsymbol, "24000")
    result = await oms.submit(
        _intent(FUT.tradingsymbol, Side.BUY, 75, product=Product.MIS),
        BracketSpec(stop_loss=Decimal("23900"), target=Decimal("24200")),
    )
    await oms.process()
    assert len(venue._resting) == 2  # SL + target resting
    quotes.set(FUT.tradingsymbol, "24050")
    assert await oms.square_off(FUT.tradingsymbol) == 1
    await oms.process()
    assert venue._resting == []
    (trade,) = oms.trades
    assert trade.exit_reason is ExitReason.SQUARE_OFF
    assert trade.pnl_gross == Decimal("3750")
    assert oms.bracket_state(result.bracket_id) == "CLOSED"
    assert venue.blocked_margin() == 0


async def test_engine_trades_lots_on_nfo_and_settles_expiry():
    clock = {"now": MON}
    venue, quotes = _venue(clock=lambda: clock["now"])
    oms = OrderManagementSystem(venue, lambda: clock["now"])
    strategy = Strategy(
        name="test",
        rule={"op": "GT", "left": {"feature": "close"}, "right": {"const": 0}},
        product=Product.CNC,  # maps to NRML for F&O
        exit=ExitRule(),
        quantity=2,  # lots
    )
    engine = PaperTradingEngine(
        ACCOUNT, venue, oms, RiskGate(), RiskProfile(), TradingCalendar(), lambda: clock["now"], [strategy],
        contracts=venue._contracts,
    )  # fmt: skip
    quotes.set(CE.tradingsymbol, "100")
    quote = MarketQuote(CE.tradingsymbol, Decimal("100"))
    signal = await engine.process_tick(CE.tradingsymbol, quote, {"close": Decimal("100")})
    assert signal is not None
    (pos,) = await venue.positions()
    assert pos.net_qty == 150 and pos.product is Product.NRML
    assert venue.fills[0].qty == 150

    clock["now"] = datetime(2026, 1, 27, 15, 45)
    fills = await engine.settle_expiries(EXPIRY, {"NIFTY": Decimal("24200")})
    assert len(fills) == 1
    (trade,) = oms.trades
    assert trade.exit_reason is ExitReason.EXPIRY
    assert trade.exit_price == Decimal("200")
    assert trade.pnl_gross == Decimal("15000")  # (200 - 100) x 150


# ---------------------------------------------------------------- Zerodha


def test_kite_params_for_fno_order():
    params = intent_to_kite_params(_intent(FUT.tradingsymbol, Side.BUY, 75))
    assert params["exchange"] == "NFO" and params["product"] == "NRML" and params["quantity"] == 75


async def test_zerodha_positions_keep_nrml():
    class Kite:
        def positions(self):
            return {"net": [{"tradingsymbol": "NIFTY26JANFUT", "product": "NRML", "quantity": 75,
                             "average_price": 24000, "realised": 0, "last_price": 24010}]}  # fmt: skip

    (pos,) = await ZerodhaAdapter(Kite(), "k", "s", ACCOUNT).get_positions()
    assert pos.product is Product.NRML


def test_parse_kite_quote_uses_depth_and_circuits():
    raw = {
        "last_price": 101.5,
        "volume": 123456,
        "depth": {"buy": [{"price": 101.45, "quantity": 75}], "sell": [{"price": 101.6, "quantity": 150}]},
        "upper_circuit_limit": 400.0,
        "lower_circuit_limit": 0.05,
    }
    q = parse_kite_quote("X", raw)
    assert (q.ltp, q.bid, q.ask, q.day_volume) == (Decimal("101.5"), Decimal("101.45"), Decimal("101.6"), 123456)
    assert q.upper_band == Decimal("400.0")
    empty = parse_kite_quote("X", {"last_price": 5, "depth": {"buy": [{"price": 0}], "sell": []}})
    assert empty.bid is None and empty.ask is None


async def test_kite_quote_source_routes_fno_symbols_to_nfo():
    class Kite:
        def quote(self, keys):
            self.keys = keys
            return {k: {"last_price": 10} for k in keys}

    kite = Kite()
    src = KiteQuoteSource(kite, ContractRegistry([FUT]))
    quotes = await src.get_quotes([FUT.tradingsymbol, "INFY"])
    assert kite.keys == ["NFO:NIFTY26JANFUT", "NSE:INFY"]
    assert quotes["INFY"].ltp == Decimal("10")


def test_read_only_kite_blocks_order_api():
    class Kite:
        def place_order(self, **kw):
            raise AssertionError("must never be reached")

        def quote(self, keys):
            return {}

    ro = ReadOnlyKite(Kite())
    assert ro.quote([]) == {}
    for name in ("place_order", "modify_order", "cancel_order", "place_gtt", "exit_order"):
        with pytest.raises(AttributeError):
            getattr(ro, name)


# ---------------------------------------------------------------- full session (simulated feed)


def test_streaming_indicators_warm_up():
    ind = StreamingIndicators()
    feats = [ind.update(Decimal(100 + i)) for i in range(25)]
    assert "EMA_9" not in feats[7] and "EMA_9" in feats[8]
    assert "EMA_21" in feats[20]
    assert "RSI_14" not in feats[13] and feats[14]["RSI_14"] == Decimal(100)  # only up-moves


class FakeKite:
    """Market-data-only fake. Price path per minute since 09:15 (synthetic):
    falls for 30 min, rallies for 60, then drifts down - one clean EMA
    crossover for the strategy to act on."""

    def __init__(self, clock):
        self.clock = clock
        self.order_calls = 0

    def _minute(self):
        start = self.clock().replace(hour=9, minute=15, second=0, microsecond=0)
        return int((self.clock() - start).total_seconds() // 60)

    def _price(self):
        m = self._minute()
        if m < 30:
            return 24000 - 5 * m
        if m < 90:
            return 23850 + 8 * (m - 30)
        return 24330 - 3 * (m - 90)

    def instruments(self, exchange):
        return KITE_ROWS

    def ltp(self, keys):
        return {k: {"last_price": 24010} for k in keys}

    def quote(self, keys):
        return {k: {"last_price": self._price(), "volume": 5_000_000} for k in keys}

    def place_order(self, **kw):  # pragma: no cover - ReadOnlyKite must block this
        self.order_calls += 1
        raise AssertionError("paper session tried to place a real order")


async def test_full_intraday_futures_session_paper_only():
    now = {"t": IST.localize(datetime(2026, 1, 5, 9, 10))}  # starts pre-open, waits for 09:15

    def clock():
        return now["t"]

    async def sleep(seconds):
        now["t"] += timedelta(seconds=seconds)

    kite = FakeKite(clock)
    logs: list[str] = []
    report = await run_session(
        kite,
        SessionConfig(underlying="NIFTY", instrument="FUT", lots=1, capital=Decimal("500000"), poll_seconds=15),
        clock=clock,
        sleep=sleep,
        log=logs.append,
    )
    assert kite.order_calls == 0
    assert report.contract.tradingsymbol == "NIFTY26JANFUT"
    assert report.trades, "expected the crossover to produce at least one paper trade"
    first = report.trades[0]
    assert first.qty == 75
    assert first.exit_reason in (ExitReason.TARGET, ExitReason.STOP_LOSS, ExitReason.SQUARE_OFF)
    # Book is flat at the end and P&L reconciles to trades after charges.
    assert len(report.fills) == 2 * len(report.trades)
    assert report.net_pnl == sum((t.pnl_net for t in report.trades), Decimal(0))
    assert now["t"].time() >= datetime(2026, 1, 5, 15, 15).time()
    assert any("ENTRY" in line for line in logs)
    assert "PAPER TRADING - no real orders were placed." in report.lines()


async def test_contracts_added_after_construction_are_seen():
    """Regression: an EMPTY registry is falsy - `contracts or ContractRegistry()`
    used to swap in a fresh one, so later-added contracts were 'unknown'."""
    registry = ContractRegistry()
    quotes = Quotes()
    venue = PaperExecutionVenue(
        ACCOUNT, quotes, TradingCalendar(), lambda: MON, starting_cash=Decimal("500000"),
        slippage=NO_SLIP, contracts=registry,
    )  # fmt: skip
    registry.add(FUT)
    quotes.set(FUT.tradingsymbol, "24000")
    assert (await venue.place(_intent(FUT.tradingsymbol, Side.BUY, 75))).state is OrderState.COMPLETE
