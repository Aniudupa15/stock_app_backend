"""PaperTradingEngine - the end-to-end paper loop (Phase 6 default mode).

One call, `process_tick`, does the full cycle for a symbol:
  1. advance existing orders (fill resting entries/children, run OCO, trail SL)
  2. if flat and a strategy fires -> risk-gate the entry -> submit a bracket

F&O: pass a `ContractRegistry` and any symbol in it is traded as a
derivative - `Strategy.quantity` then means LOTS (converted to units with the
contract's lot size), the order goes to the contract's exchange (NFO), a CNC
strategy maps to NRML, and the risk gate sizes on margin rather than notional.

No broker code anywhere: swap the `PaperExecutionVenue` for a
`BrokerExecutionVenue` and the identical loop trades live (Phase 2 P1).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal
from uuid import UUID, uuid4

from libs.derivatives.contracts import ContractRegistry
from libs.derivatives.margin import FnoMarginModel
from libs.engine.risk_state import RiskStateBuilder
from libs.oms.oms import OrderManagementSystem
from libs.risk.gate import RiskGate, RiskProfile
from libs.strategy.engine import Strategy, StrategyEvaluator
from libs.trading_calendar.calendar import TradingCalendar
from libs.trading_domain.entities import BracketSpec, Fill, MarketQuote, OrderIntent, Signal
from libs.trading_domain.enums import OrderType, Product, Side
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
        contracts: ContractRegistry | None = None,
        margin_model: FnoMarginModel | None = None,
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
        self._contracts = contracts if contracts is not None else ContractRegistry()
        self._margin = margin_model or FnoMarginModel()

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

            contract = self._contracts.get(symbol)
            quantity, product, exchange = strategy.quantity, strategy.product, "NSE"
            lot_size, funds_per_unit = 1, None
            if contract is not None:
                lot_size = contract.lot_size
                quantity = strategy.quantity * lot_size
                product = Product.NRML if strategy.product is Product.CNC else strategy.product
                exchange = contract.exchange
                sign = 1 if strategy.side is Side.BUY else -1
                funds_per_unit = self._margin.entry_cost_per_unit(contract, sign, quote.ltp)
            intent = OrderIntent(
                intent_id=uuid4(),
                account_id=self._account_id,
                symbol=symbol,
                side=strategy.side,
                order_type=OrderType.MARKET,
                product=product,
                quantity=quantity,
                exchange=exchange,
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
                lot_size=lot_size,
                funds_per_unit=funds_per_unit,
            )
            if not verdict.allowed:
                continue
            intent = replace(intent, quantity=verdict.quantity)

            spec = BracketSpec(
                stop_loss=signal.stop_loss,
                target=signal.targets[0] if signal.targets else None,
                trailing=strategy.exit.trailing,
                breakeven_pct=strategy.exit.breakeven_pct,
            )
            result = await self._oms.submit(intent, spec)
            await self._oms.process()  # place children as soon as the entry fills
            if result.accepted and result.bracket_id is not None:
                self._active[symbol] = result.bracket_id
                return signal
        return None

    async def square_off(self, symbol: str) -> int:
        """Flatten `symbol` at market (intraday MIS square-off / manual exit)."""
        sent = await self._oms.square_off(symbol)
        await self._oms.process()
        return sent

    async def settle_expiries(self, as_of: date, underlying_prices: dict[str, Decimal]) -> list[Fill]:
        """Settle expired F&O positions at exchange settlement value and
        journal them as EXPIRY trades. Paper venue only."""
        fills = self._venue.settle_expired(as_of, underlying_prices)
        await self._oms.process()  # drain cancellations of resting children
        for fill in fills:
            self._oms.close_by_settlement(fill)
            self._active.pop(fill.symbol, None)
        return fills
