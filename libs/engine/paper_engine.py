"""PaperTradingEngine - the end-to-end paper loop (Phase 6 default mode).

One call, `process_tick`, does the full cycle for a symbol:
  1. advance existing orders (fill resting entries/children, run OCO, trail SL)
  2. if flat and a strategy fires -> risk-gate the entry -> submit a bracket

No broker code anywhere: swap the `PaperExecutionVenue` for a
`BrokerExecutionVenue` and the identical loop trades live (Phase 2 P1).
"""

from __future__ import annotations

from dataclasses import replace
from uuid import UUID, uuid4

from libs.engine.risk_state import RiskStateBuilder
from libs.oms.oms import OrderManagementSystem
from libs.risk.gate import RiskGate, RiskProfile
from libs.strategy.engine import Strategy, StrategyEvaluator
from libs.trading_calendar.calendar import TradingCalendar
from libs.trading_domain.entities import BracketSpec, MarketQuote, OrderIntent, Signal
from libs.trading_domain.enums import OrderType
from libs.trading_domain.ports import ExecutionVenuePort


class PaperTradingEngine:
    def __init__(
        self,
        account_id: UUID,
        venue: ExecutionVenuePort,
        oms: OrderManagementSystem,
        risk_gate: RiskGate,
        profile: RiskProfile,
        calendar: TradingCalendar,
        clock,
        strategies: list[Strategy],
        evaluator: StrategyEvaluator | None = None,
    ) -> None:
        self._account_id = account_id
        self._venue = venue
        self._oms = oms
        self._risk_gate = risk_gate
        self._profile = profile
        self._calendar = calendar
        self._clock = clock
        self._strategies = strategies
        self._evaluator = evaluator or StrategyEvaluator()
        self._state_builder = RiskStateBuilder(venue, oms, clock)
        self._active: dict[str, str] = {}  # symbol -> live bracket_id

    async def process_tick(
        self,
        symbol: str,
        quote: MarketQuote,
        features: dict | None = None,
        prev_features: dict | None = None,
    ) -> Signal | None:
        # 1. Advance existing orders: fill resting, OCO, trailing.
        await self._venue.on_tick(quote)
        await self._oms.process()
        await self._oms.on_tick(quote)

        bracket_id = self._active.get(symbol)
        if bracket_id is not None and self._oms.bracket_state(bracket_id) == "CLOSED":
            self._active.pop(symbol, None)

        # 2. Entry: only when flat on this symbol and no live bracket.
        if features is None or symbol in self._active:
            return None
        state = await self._state_builder.build(symbol)
        if state.current_net_qty != 0:
            return None

        for strategy in self._strategies:
            signal = self._evaluator.evaluate(strategy, symbol, features, prev_features, quote.ltp, self._clock())
            if signal is None:
                continue

            intent = OrderIntent(
                intent_id=uuid4(),
                account_id=self._account_id,
                symbol=symbol,
                side=strategy.side,
                order_type=OrderType.MARKET,
                product=strategy.product,
                quantity=strategy.quantity,
                strategy_id=strategy.strategy_id,
                signal_id=None,
            )
            verdict = self._risk_gate.evaluate(
                intent,
                state=state,
                profile=self._profile,
                now=self._clock(),
                calendar=self._calendar,
                entry_price=quote.ltp,
                stop_loss=signal.stop_loss,
            )
            if not verdict.allowed:
                continue
            intent = replace(intent, quantity=verdict.quantity)

            spec = BracketSpec(
                stop_loss=signal.stop_loss,
                target=signal.targets[0] if signal.targets else None,
                trailing=strategy.exit.trailing,
            )
            result = await self._oms.submit(intent, spec)
            await self._oms.process()  # place children as soon as the entry fills
            if result.accepted and result.bracket_id is not None:
                self._active[symbol] = result.bracket_id
                return signal
        return None
