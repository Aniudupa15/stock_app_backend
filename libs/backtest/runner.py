"""Backtester - replays historical bars through the SAME components that run
paper and live: PaperExecutionVenue + OMS + RiskGate + StrategyEvaluator
(Phase 2 P1 - what you backtest is what you paper-trade is what you trade).

Intrabar fill convention (documented, conservative for long strategies): each
bar is walked as open -> low -> high -> close. Feeding the low before the high
means a long position's stop is checked before its target within the same bar
(pessimistic). Entry signals are evaluated on the bar close and fill at the
close, so no future information leaks into a decision.

Fill bias is deliberately conservative ("never rosier", Phase 1 §5): a LIMIT
target fills at its limit price (not the better intrabar extreme), while a
stop-MARKET child fills at the intrabar extreme it triggered on (worse than
the trigger). Trigger-price / gap-aware stop fills are a later refinement.

v1 scope: single symbol, one open bracket at a time. Multi-symbol portfolio
backtests are a later extension.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, time
from decimal import Decimal
from uuid import UUID, uuid4

from libs.backtest.bar import Bar
from libs.backtest.metrics import BacktestMetrics, compute_metrics
from libs.engine.risk_state import RiskStateBuilder
from libs.execution.paper import PaperExecutionVenue
from libs.execution.slippage import SlippageModel
from libs.oms.oms import OrderManagementSystem
from libs.risk.gate import RiskGate, RiskProfile
from libs.strategy.engine import Strategy, StrategyEvaluator
from libs.trading_calendar.calendar import TradingCalendar
from libs.trading_domain.entities import BracketSpec, MarketQuote, OrderIntent, Trade
from libs.trading_domain.enums import OrderType

# (index) -> (features, prev_features): computed by the caller from indicators
# over bars[:index+1], so the backtester stays agnostic to how features are made.
FeatureBuilder = Callable[[int], tuple[dict, dict | None]]

_SESSION_TIME = time(10, 0)  # keeps the calendar's is_open() True on trading days


@dataclass(frozen=True, slots=True)
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[tuple[datetime, Decimal]]
    metrics: BacktestMetrics
    starting_cash: Decimal
    final_equity: Decimal


class Backtester:
    def __init__(
        self,
        *,
        symbol: str,
        strategy: Strategy,
        starting_cash: Decimal,
        calendar: TradingCalendar | None = None,
        slippage: SlippageModel | None = None,
        profile: RiskProfile | None = None,
        account_id: UUID | None = None,
    ) -> None:
        self._symbol = symbol
        self._strategy = strategy
        self._starting_cash = starting_cash
        self._calendar = calendar or TradingCalendar()
        self._slippage = slippage
        self._profile = profile or RiskProfile()
        self._account_id = account_id or uuid4()

    async def run(self, bars: list[Bar], feature_builder: FeatureBuilder) -> BacktestResult:
        now_holder: dict[str, datetime] = {}
        quote_holder: dict[str, MarketQuote] = {}
        clock = lambda: now_holder["now"]  # noqa: E731

        class _Quotes:
            async def get_quote(self, symbol: str) -> MarketQuote:
                return quote_holder["quote"]

        venue = PaperExecutionVenue(
            self._account_id,
            _Quotes(),
            self._calendar,
            clock,
            starting_cash=self._starting_cash,
            slippage=self._slippage,
        )
        oms = OrderManagementSystem(venue, clock)
        risk_gate = RiskGate()
        evaluator = StrategyEvaluator()
        state_builder = RiskStateBuilder(venue, oms, clock)

        active_bracket: str | None = None
        equity_curve: list[tuple[datetime, Decimal]] = []

        for i, bar in enumerate(bars):
            now = datetime.combine(bar.day, _SESSION_TIME)
            now_holder["now"] = now
            if not self._calendar.is_open(now):
                continue

            # 1. Advance existing orders across the bar's intrabar path.
            for px in (bar.open, bar.low, bar.high):
                q = MarketQuote(symbol=self._symbol, ltp=px, day_volume=bar.volume)
                quote_holder["quote"] = q
                await venue.on_tick(q)
                await oms.process()
                await oms.on_tick(q)
            if active_bracket is not None and oms.bracket_state(active_bracket) == "CLOSED":
                active_bracket = None

            # 2. Entry evaluation on the bar close.
            close_q = MarketQuote(symbol=self._symbol, ltp=bar.close, day_volume=bar.volume)
            quote_holder["quote"] = close_q
            if active_bracket is None:
                state = await state_builder.build(self._symbol)
                if state.current_net_qty == 0:
                    features, prev = feature_builder(i)
                    signal = evaluator.evaluate(self._strategy, self._symbol, features, prev, bar.close, now)
                    if signal is not None:
                        intent = OrderIntent(
                            intent_id=uuid4(),
                            account_id=self._account_id,
                            symbol=self._symbol,
                            side=self._strategy.side,
                            order_type=OrderType.MARKET,
                            product=self._strategy.product,
                            quantity=self._strategy.quantity,
                            strategy_id=self._strategy.strategy_id,
                        )
                        verdict = risk_gate.evaluate(
                            intent,
                            state=state,
                            profile=self._profile,
                            now=now,
                            calendar=self._calendar,
                            entry_price=bar.close,
                            stop_loss=signal.stop_loss,
                        )
                        if verdict.allowed:
                            intent = replace(intent, quantity=verdict.quantity)
                            spec = BracketSpec(
                                stop_loss=signal.stop_loss,
                                target=signal.targets[0] if signal.targets else None,
                                trailing=self._strategy.exit.trailing,
                            )
                            res = await oms.submit(intent, spec)
                            await oms.process()
                            if res.accepted:
                                active_bracket = res.bracket_id

            equity_curve.append((now, venue.equity()))

        metrics = compute_metrics(oms.trades, equity_curve, self._starting_cash)
        return BacktestResult(
            trades=list(oms.trades),
            equity_curve=equity_curve,
            metrics=metrics,
            starting_cash=self._starting_cash,
            final_equity=venue.equity(),
        )
