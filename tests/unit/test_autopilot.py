"""Auto-pilot: autonomous multi-symbol paper trading + EOD report. Pure."""

from datetime import date, datetime
from decimal import Decimal
from uuid import uuid4

from libs.engine.paper_engine import PaperTradingEngine
from libs.execution.paper import PaperExecutionVenue
from libs.execution.slippage import SlippageModel
from libs.oms.oms import OrderManagementSystem
from libs.risk.gate import RiskGate, RiskProfile
from libs.trading_calendar.calendar import TradingCalendar
from libs.trading_domain.entities import MarketQuote
from services.trading_service.autopilot.engine import AutoPilot
from services.trading_service.autopilot.eod_report import build_eod_report
from services.trading_service.autopilot.strategies import default_momentum_strategy

MON = datetime(2026, 1, 5, 10, 0)
DAY = date(2026, 1, 5)
NO_SLIP = SlippageModel(base_bps=Decimal("0"), impact_coeff_bps=Decimal("0"), illiquid_penalty_bps=Decimal("0"))
ACCOUNT = uuid4()

MOMENTUM = {"close": 105, "EMA_20": 100, "EMA_50": 98, "RSI_14": 60}  # passes the default rule
WEAK = {"close": 95, "EMA_20": 100, "EMA_50": 98, "RSI_14": 40}  # fails (close < EMA20)


class FakeQuotes:
    def __init__(self):
        self.quotes: dict[str, MarketQuote] = {}

    async def get_quote(self, symbol: str) -> MarketQuote:
        return self.quotes[symbol]


def _q(symbol, ltp):
    return MarketQuote(symbol=symbol, ltp=Decimal(str(ltp)), day_volume=1_000_000)


def _autopilot():
    quotes = FakeQuotes()
    venue = PaperExecutionVenue(
        ACCOUNT, quotes, TradingCalendar(), lambda: MON, starting_cash=Decimal("10000000"), slippage=NO_SLIP
    )
    oms = OrderManagementSystem(venue, lambda: MON)
    engine = PaperTradingEngine(
        ACCOUNT,
        venue,
        oms,
        RiskGate(),
        RiskProfile(),
        TradingCalendar(),
        lambda: MON,
        [default_momentum_strategy(quantity=10)],
    )
    return AutoPilot(engine), venue, oms, quotes


async def test_autopilot_enters_only_the_momentum_stock():
    ap, venue, oms, quotes = _autopilot()
    quotes.quotes = {"AAA": _q("AAA", 105), "BBB": _q("BBB", 95)}
    fired = await ap.run_cycle(
        {
            "AAA": (_q("AAA", 105), MOMENTUM, None),
            "BBB": (_q("BBB", 95), WEAK, None),
        }
    )
    assert [s.symbol for s in fired] == ["AAA"]  # only the momentum stock entered
    positions = {p.symbol: p for p in await venue.positions()}
    assert positions["AAA"].net_qty == 10
    assert "BBB" not in positions


async def test_autopilot_full_day_enter_then_target_exit_and_eod_report():
    ap, venue, oms, quotes = _autopilot()
    # Morning: enter AAA on momentum.
    quotes.quotes = {"AAA": _q("AAA", 105)}
    await ap.run_cycle({"AAA": (_q("AAA", 105), MOMENTUM, None)})
    assert (await venue.positions())[0].net_qty == 10

    # Later: AAA rallies past the +3% target (108.15) -> auto-exit.
    # features=None means "manage existing positions, evaluate no new entries"
    # this cycle (avoids immediately re-buying what we just sold).
    quotes.quotes = {"AAA": _q("AAA", 108.5)}
    await ap.run_cycle({"AAA": (_q("AAA", 108.5), None, None)})

    assert len(oms.trades) == 1
    assert oms.trades[0].symbol == "AAA"
    assert (await venue.positions())[0].net_qty == 0  # flat again

    report = build_eod_report(oms.trades, DAY)
    assert report.total_trades == 1
    assert report.wins == 1
    assert report.win_rate == 1.0
    assert report.net_pnl == oms.trades[0].pnl_net
    assert report.best_symbol == "AAA"


def test_eod_report_empty_day_is_safe():
    report = build_eod_report([], DAY)
    assert report.total_trades == 0
    assert report.win_rate == 0.0
    assert report.net_pnl == Decimal("0")
    assert report.best_symbol is None


def test_default_strategy_is_white_box_and_inspectable():
    strat = default_momentum_strategy()
    assert strat.rule["op"] == "AND"
    assert strat.exit.target_pct == Decimal("3")
    assert strat.exit.stop_loss_pct == Decimal("1.5")
