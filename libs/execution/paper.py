"""PaperExecutionVenue - fills orders against live/last quotes with realistic
slippage and the exact NSE charge schedule, tracking a virtual cash balance
and positions. Behaves like a broker venue but never places a real order.

Deliberately handles only single orders. Brackets/OCO/trailing-SL are the
OMS's job (Phase 3 §4.2) layered on top of this venue, so this same code runs
identically under paper and live.

Scope of v1 (documented, not hidden):
  * MARKET fully fills at ref +/- slippage. Partial-fill-by-depth is a future
    refinement; size shows up as wider slippage instead.
  * LIMIT / SL / SL_M rest until a tick crosses their price/trigger, then fill
    via `on_tick(quote)` (the engine's paper loop feeds ticks in).
  * Equity cash is tracked at full notional (no intraday leverage). Margin
    sufficiency for equity is the risk engine's concern, not the venue's.

F&O (when a `ContractRegistry` is supplied and the intent's exchange is NFO/BFO):
  * Orders must be whole lots, on a known, unexpired contract, product MIS/NRML.
  * Futures are margined, not paid for: no notional changes hands; realised
    P&L and charges hit cash, unrealised MTM counts toward available funds.
  * Options are premium-settled: a BUY pays the premium, a SELL receives it.
  * Opening orders are rejected (like a broker's RMS would) if the margin
    proxy in `FnoMarginModel` exceeds available funds.
  * `settle_expired()` closes positions in expired contracts at their
    settlement value (futures -> underlying, options -> intrinsic).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from libs.charges import nse_fno
from libs.charges.models import Charges, Side
from libs.charges.nse_equity import DEFAULT_NSE_EQUITY_SCHEDULE, ChargeSchedule, compute
from libs.derivatives.contracts import FNO_EXCHANGES, ContractRegistry, FnoContract, InstrumentKind
from libs.derivatives.margin import FnoMarginModel
from libs.execution.slippage import SlippageModel
from libs.trading_calendar.calendar import TradingCalendar
from libs.trading_domain.entities import (
    Fill,
    Margin,
    MarketQuote,
    OrderAck,
    OrderEvent,
    OrderIntent,
    Position,
)
from libs.trading_domain.enums import OrderState, OrderType, Product
from libs.trading_domain.ports import EventSink, ExecutionVenuePort, QuoteSourcePort

# Clock is injected so tests and backtests control "now".
Clock = Callable[[], datetime]


@dataclass
class _MutablePosition:
    net_qty: int = 0
    avg_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")


@dataclass
class _RestingOrder:
    order_id: str
    intent: OrderIntent


@dataclass
class _Result:
    """Internal fill outcome (also what the round-trip P&L is derived from)."""

    fills: list[Fill] = field(default_factory=list)


class PaperExecutionVenue(ExecutionVenuePort):
    def __init__(
        self,
        account_id: UUID,
        quote_source: QuoteSourcePort,
        calendar: TradingCalendar,
        clock: Clock,
        *,
        starting_cash: Decimal,
        schedule: ChargeSchedule = DEFAULT_NSE_EQUITY_SCHEDULE,
        slippage: SlippageModel | None = None,
        contracts: ContractRegistry | None = None,
        fno_schedule: nse_fno.FnoChargeSchedule = nse_fno.DEFAULT_NSE_FNO_SCHEDULE,
        margin_model: FnoMarginModel | None = None,
    ) -> None:
        self._account_id = account_id
        self._quotes = quote_source
        self._calendar = calendar
        self._clock = clock
        self._cash = starting_cash
        self._starting_cash = starting_cash
        self._schedule = schedule
        self._slippage = slippage or SlippageModel()
        self._contracts = contracts if contracts is not None else ContractRegistry()
        self._fno_schedule = fno_schedule
        self._margin = margin_model or FnoMarginModel()
        self._settle_counter = 0
        self._positions: dict[tuple[str, str], _MutablePosition] = {}
        self._resting: list[_RestingOrder] = []
        self._last_price: dict[str, Decimal] = {}
        self._event_sink: EventSink | None = None
        self._counter = 0
        self.fills: list[Fill] = []  # audit log of every executed fill

    # --- ExecutionVenuePort ---

    def set_event_sink(self, sink: EventSink) -> None:
        self._event_sink = sink

    async def place(self, intent: OrderIntent) -> OrderAck:
        now = self._clock()
        if not self._calendar.is_open(now):
            return self._reject(intent, "market closed")

        quote = await self._quotes.get_quote(intent.symbol)
        self._last_price[intent.symbol] = quote.ltp
        order_id = self._next_id()

        if self._circuit_blocks(intent, quote):
            return self._reject(intent, "circuit locked", order_id=order_id)

        if intent.exchange in FNO_EXCHANGES:
            problem = self._fno_problem(intent, quote, now.date())
            if problem:
                return self._reject(intent, problem, order_id=order_id)

        if intent.order_type is OrderType.MARKET:
            price = self._slippage.adjust(
                intent.side, self._reference(intent.side, quote), intent.quantity, quote.day_volume
            )
            self._execute(order_id, intent, intent.quantity, price, now)
            return OrderAck(intent.intent_id, order_id, OrderState.COMPLETE)

        if intent.order_type is OrderType.LIMIT and self._limit_marketable(intent, quote):
            # Marketable limit: fill at the limit price (never worse than it).
            self._execute(order_id, intent, intent.quantity, intent.price, now)
            return OrderAck(intent.intent_id, order_id, OrderState.COMPLETE)

        # LIMIT (not marketable) or SL / SL_M -> rest until a tick crosses.
        self._resting.append(_RestingOrder(order_id, intent))
        self._emit(OrderEvent(order_id, intent.intent_id, OrderState.OPEN, 0, intent.quantity, Decimal("0"), now))
        return OrderAck(intent.intent_id, order_id, OrderState.OPEN)

    async def cancel(self, venue_order_id: str) -> OrderAck:
        for r in self._resting:
            if r.order_id == venue_order_id:
                self._resting.remove(r)
                self._emit(
                    OrderEvent(
                        venue_order_id, r.intent.intent_id, OrderState.CANCELLED, 0, 0, Decimal("0"), self._clock()
                    )
                )
                return OrderAck(r.intent.intent_id, venue_order_id, OrderState.CANCELLED)
        return OrderAck(
            intent_id=None, venue_order_id=venue_order_id, state=OrderState.REJECTED, reason="unknown order"
        )

    async def positions(self) -> list[Position]:
        out: list[Position] = []
        for (symbol, product), pos in self._positions.items():
            if pos.net_qty == 0 and pos.realized_pnl == 0:
                continue
            out.append(
                Position(
                    account_id=self._account_id,
                    symbol=symbol,
                    product=_product_from_key(product),
                    net_qty=pos.net_qty,
                    avg_price=pos.avg_price,
                    realized_pnl=pos.realized_pnl,
                    ltp=self._last_price.get(symbol),
                )
            )
        return out

    async def available_margin(self) -> Margin:
        return Margin(available=self._cash + self._futures_mtm() - self.blocked_margin())

    # --- paper-specific: tick-driven resting-order fills ---

    async def on_tick(self, quote: MarketQuote) -> None:
        """Feed a tick; fill any resting order whose price/trigger is now met."""
        self._last_price[quote.symbol] = quote.ltp
        now = self._clock()
        remaining: list[_RestingOrder] = []
        for r in self._resting:
            if r.intent.symbol != quote.symbol or not self._resting_triggers(r.intent, quote):
                remaining.append(r)
                continue
            price = self._resting_fill_price(r.intent, quote)
            self._execute(r.order_id, r.intent, r.intent.quantity, price, now)
        self._resting = remaining

    # --- helpers ---

    def cash(self) -> Decimal:
        return self._cash

    def get_fill(self, order_id: str) -> Fill | None:
        """Exact recorded fill (incl. charges) for an order - used by the OMS
        to build round-trip Trades with precise costs."""
        for fill in self.fills:
            if fill.order_id == order_id:
                return fill
        return None

    def equity(self) -> Decimal:
        """Cash + mark-to-market of open positions at last seen prices.
        Futures contribute only their unrealised P&L (no notional was paid)."""
        mtm = Decimal("0")
        for (symbol, _), pos in self._positions.items():
            ltp = self._last_price.get(symbol)
            if ltp is None or self._is_future(symbol):
                continue
            mtm += Decimal(pos.net_qty) * ltp
        return self._cash + mtm + self._futures_mtm()

    def blocked_margin(self) -> Decimal:
        """Margin held against open F&O positions (paper proxy)."""
        total = Decimal("0")
        for (symbol, _), pos in self._positions.items():
            contract = self._contracts.get(symbol)
            if contract is None or pos.net_qty == 0:
                continue
            price = self._last_price.get(symbol, pos.avg_price)
            total += self._margin.blocked(contract, pos.net_qty, price)
        return total

    def settle_expired(self, as_of: date, underlying_prices: dict[str, Decimal]) -> list[Fill]:
        """Settle every open position in a contract that expired before
        `as_of` (or on it - call after the close on expiry day). Resting
        orders in those contracts are cancelled. `underlying_prices` maps
        underlying -> final settlement price; positions whose underlying is
        missing are left open (and reported by the caller)."""
        settled: list[Fill] = []
        now = self._clock()
        for (symbol, product), pos in list(self._positions.items()):
            contract = self._contracts.get(symbol)
            if contract is None or pos.net_qty == 0 or contract.expiry > as_of:
                continue
            spot = underlying_prices.get(contract.underlying)
            if spot is None:
                continue
            for r in [r for r in self._resting if r.intent.symbol == symbol]:
                self._resting.remove(r)
                self._emit(OrderEvent(r.order_id, r.intent.intent_id, OrderState.CANCELLED, 0, 0, Decimal("0"), now))
            price = contract.settlement_price(spot)
            qty = abs(pos.net_qty)
            side = Side.SELL if pos.net_qty > 0 else Side.BUY
            charges = nse_fno.compute_settlement(contract.kind, pos.net_qty, price, schedule=self._fno_schedule)
            self._settle_counter += 1
            order_id = f"SETTLE-{self._settle_counter:06d}"
            self._apply_cash(contract, side, qty, price, charges, symbol, Product(product))
            self._last_price[symbol] = price
            fill = Fill(order_id, None, symbol, side, qty, price, charges, now)
            self.fills.append(fill)
            settled.append(fill)
        return settled

    def _next_id(self) -> str:
        self._counter += 1
        return f"PAPER-{self._counter:06d}"

    def _emit(self, event: OrderEvent) -> None:
        if self._event_sink is not None:
            self._event_sink(event)

    def _reject(self, intent: OrderIntent, reason: str, *, order_id: str | None = None) -> OrderAck:
        oid = order_id or self._next_id()
        self._emit(
            OrderEvent(
                oid, intent.intent_id, OrderState.REJECTED, 0, intent.quantity, Decimal("0"), self._clock(), reason
            )
        )
        return OrderAck(intent.intent_id, None, OrderState.REJECTED, reason)

    @staticmethod
    def _reference(side: Side, quote: MarketQuote) -> Decimal:
        if side is Side.BUY:
            return quote.ask or quote.ltp
        return quote.bid or quote.ltp

    @staticmethod
    def _limit_marketable(intent: OrderIntent, quote: MarketQuote) -> bool:
        if intent.price is None:
            return False
        if intent.side is Side.BUY:
            return quote.ltp <= intent.price
        return quote.ltp >= intent.price

    @staticmethod
    def _circuit_blocks(intent: OrderIntent, quote: MarketQuote) -> bool:
        if intent.side is Side.BUY and quote.upper_band is not None and quote.ltp >= quote.upper_band:
            return True
        if intent.side is Side.SELL and quote.lower_band is not None and quote.ltp <= quote.lower_band:
            return True
        return False

    @staticmethod
    def _resting_triggers(intent: OrderIntent, quote: MarketQuote) -> bool:
        if intent.order_type is OrderType.LIMIT:
            if intent.side is Side.BUY:
                return intent.price is not None and quote.ltp <= intent.price
            return intent.price is not None and quote.ltp >= intent.price
        # SL / SL_M: trigger when price moves through the stop.
        trig = intent.trigger_price
        if trig is None:
            return False
        if intent.side is Side.BUY:
            return quote.ltp >= trig
        return quote.ltp <= trig

    def _resting_fill_price(self, intent: OrderIntent, quote: MarketQuote) -> Decimal:
        if intent.order_type in (OrderType.LIMIT, OrderType.SL) and intent.price is not None:
            return intent.price  # limit fill, at the specified price
        # SL_M -> market with slippage from the reference.
        return self._slippage.adjust(
            intent.side, self._reference(intent.side, quote), intent.quantity, quote.day_volume
        )

    def _execute(self, order_id: str, intent: OrderIntent, qty: int, price: Decimal, now: datetime) -> None:
        contract = self._fno_contract(intent)
        if contract is not None:
            charges = nse_fno.compute(intent.side, contract.kind, qty, price, schedule=self._fno_schedule)
        else:
            charges = compute(intent.side, intent.product, qty, price, schedule=self._schedule)
        self._apply_cash(contract, intent.side, qty, price, charges, intent.symbol, intent.product)
        fill = Fill(order_id, intent.intent_id, intent.symbol, intent.side, qty, price, charges, now)
        self.fills.append(fill)
        self._emit(OrderEvent(order_id, intent.intent_id, OrderState.COMPLETE, qty, 0, price, now))

    def _apply_cash(
        self,
        contract: FnoContract | None,
        side: Side,
        qty: int,
        price: Decimal,
        charges: Charges,
        symbol: str,
        product: Product,
    ) -> None:
        """Move cash for one executed leg and update the position."""
        if contract is not None and contract.kind is InstrumentKind.FUT:
            # Margined: only realised P&L (and charges) touch cash.
            pos = self._positions.get((symbol, product.value))
            before = pos.realized_pnl if pos else Decimal("0")
            self._update_position(symbol, product, side, qty, price)
            after = self._positions[(symbol, product.value)].realized_pnl
            self._cash += after - before - charges.total
            return
        notional = Decimal(qty) * price
        if side is Side.BUY:
            self._cash -= notional + charges.total
        else:
            self._cash += notional - charges.total
        self._update_position(symbol, product, side, qty, price)

    # --- F&O helpers ---

    def _fno_contract(self, intent: OrderIntent) -> FnoContract | None:
        if intent.exchange not in FNO_EXCHANGES:
            return None
        return self._contracts.get(intent.symbol)

    def _is_future(self, symbol: str) -> bool:
        contract = self._contracts.get(symbol)
        return contract is not None and contract.kind is InstrumentKind.FUT

    def _futures_mtm(self) -> Decimal:
        mtm = Decimal("0")
        for (symbol, _), pos in self._positions.items():
            if pos.net_qty == 0 or not self._is_future(symbol):
                continue
            ltp = self._last_price.get(symbol)
            if ltp is not None:
                mtm += (ltp - pos.avg_price) * pos.net_qty
        return mtm

    def _fno_problem(self, intent: OrderIntent, quote: MarketQuote, today: date) -> str | None:
        """Broker-RMS-style validation of an F&O order; None if acceptable."""
        contract = self._contracts.get(intent.symbol)
        if contract is None:
            return "unknown F&O contract"
        if contract.is_expired(today):
            return "contract expired"
        if intent.product is Product.CNC:
            return "CNC not allowed for F&O (use MIS or NRML)"
        if intent.quantity % contract.lot_size != 0:
            return f"quantity must be a multiple of lot size {contract.lot_size}"

        pos = self._positions.get((intent.symbol, intent.product.value))
        net = pos.net_qty if pos else 0
        signed = intent.quantity if intent.side is Side.BUY else -intent.quantity
        closing = min(abs(signed), abs(net)) if net and (net > 0) != (signed > 0) else 0
        opening = abs(signed) - closing
        if opening == 0:
            return None  # pure exit - never blocked

        ref = intent.price if intent.price is not None and intent.order_type is OrderType.LIMIT else quote.ltp
        before = self._margin.blocked(contract, net, ref)
        after = self._margin.blocked(contract, net + signed, ref)
        needed = max(Decimal("0"), after - before)
        if contract.kind.is_option and intent.side is Side.BUY:
            needed += Decimal(opening) * ref  # premium paid up front
        available = self._cash + self._futures_mtm() - self.blocked_margin()
        if needed > available:
            return f"insufficient margin: need {needed:.2f}, available {available:.2f}"
        return None

    def _update_position(self, symbol: str, product: Product, side: Side, qty: int, price: Decimal) -> None:
        key = (symbol, product.value)
        pos = self._positions.setdefault(key, _MutablePosition())
        signed = qty if side is Side.BUY else -qty

        if pos.net_qty == 0 or (pos.net_qty > 0) == (signed > 0):
            # Opening or increasing in the same direction -> weighted avg.
            new_qty = pos.net_qty + signed
            pos.avg_price = (pos.avg_price * abs(pos.net_qty) + price * qty) / Decimal(abs(new_qty))
            pos.net_qty = new_qty
            return

        # Reducing / closing / reversing.
        closing = min(qty, abs(pos.net_qty))
        if pos.net_qty > 0:  # was long, now selling
            pos.realized_pnl += (price - pos.avg_price) * closing
        else:  # was short, now buying to cover
            pos.realized_pnl += (pos.avg_price - price) * closing

        new_qty = pos.net_qty + signed
        if new_qty == 0:
            pos.net_qty = 0
            pos.avg_price = Decimal("0")
        elif (new_qty > 0) != (pos.net_qty > 0):
            # Reversed through zero -> remainder opens a fresh position at price.
            pos.net_qty = new_qty
            pos.avg_price = price
        else:
            pos.net_qty = new_qty  # avg unchanged when merely reducing


def _product_from_key(product_value: str):
    from libs.trading_domain.enums import Product

    return Product(product_value)
