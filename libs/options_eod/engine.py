"""End-of-day options backtester.

Many strategy variants run side by side over ONE pass of the data (each with
its own book), so a parameter grid costs one load of the chain files.

Conventions:
  * A Leg is `qty` lots (+ buy / - sell) of one contract.
  * Cash flows are in index POINTS per unit: selling at 100 = +100. A trade's
    P&L in points = entry cash + exit cash. Rupees = points x lot size, minus
    Zerodha F&O charges (libs.charges.nse_fno) on every leg, both sides.
  * Fills at the day's close, worse by `slippage` points per leg per side.
  * Contracts still open on/after expiry are settled at intrinsic value from
    the index close (exercise STT on long ITM legs, no brokerage).
  * A leg with no quote is valued with Black-Scholes at India VIX and counted
    in `modelled_legs`, so you can see how much of a result rests on it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from libs.charges import nse_fno
from libs.charges.models import Side
from libs.derivatives.contracts import InstrumentKind
from libs.options_eod.chain import DayChain

_KIND = {"CE": InstrumentKind.CE, "PE": InstrumentKind.PE}


def _dec(x: float) -> Decimal:
    return Decimal(str(round(x, 2)))


@dataclass(frozen=True, slots=True)
class Leg:
    expiry: date
    strike: float
    kind: str  # CE | PE
    qty: int  # lots: +1 buy, -1 sell

    def label(self) -> str:
        return f"{'+' if self.qty > 0 else '-'}{abs(self.qty)} {self.strike:g}{self.kind}"


@dataclass
class Position:
    legs: list[Leg]
    entry_date: date
    entry_cash: float  # points per unit, after slippage (credit > 0, debit < 0)
    entry_costs: float  # rupees
    tag: str = ""

    @property
    def expiry(self) -> date:
        return min(leg.expiry for leg in self.legs)


@dataclass(frozen=True, slots=True)
class EodTrade:
    entry_date: date
    exit_date: date
    legs: str
    entry_cash: float
    exit_cash: float
    points: float
    rupees_net: float
    costs: float
    reason: str
    tag: str
    modelled_legs: int


@dataclass
class Book:
    label: str
    params: dict
    position: Position | None = None
    trades: list[EodTrade] = field(default_factory=list)
    equity: list[tuple[date, float]] = field(default_factory=list)  # rupees, marked daily
    realised: float = 0.0
    modelled_legs: int = 0
    priced_legs: int = 0


class Ctx:
    """What a strategy sees and does on one day for its own book."""

    def __init__(self, day: DayChain, book: Book, lot: int, slippage: float) -> None:
        self.day, self.book, self.lot, self.slippage = day, book, lot, slippage

    @property
    def position(self) -> Position | None:
        return self.book.position

    def _px(self, leg: Leg) -> tuple[float, bool]:
        p = self.day.price(leg.expiry, leg.strike, leg.kind)
        if p is not None:
            return p, False
        return self.day.model_price(leg.expiry, leg.strike, leg.kind), True

    def _costs(self, leg: Leg, px: float, opening: bool) -> float:
        selling = (leg.qty < 0) if opening else (leg.qty > 0)
        units = self.lot * abs(leg.qty)
        price = _dec(max(px, 0.05))
        return float(nse_fno.compute(Side.SELL if selling else Side.BUY, _KIND[leg.kind], units, price).total)

    def cost_to_close(self) -> float | None:
        """Points to flatten now at the close (no slippage) - what stop and
        profit-target rules compare against the entry credit."""
        pos = self.book.position
        if pos is None:
            return None
        return sum(-leg.qty * self._px(leg)[0] for leg in pos.legs)

    def open(self, legs: list[Leg], tag: str = "") -> bool:
        """Open at today's close; refuses (False) if any leg has no real quote."""
        if self.book.position is not None:
            return False
        quotes = [self.day.price(leg.expiry, leg.strike, leg.kind) for leg in legs]
        if any(q is None for q in quotes):
            return False
        cash = costs = 0.0
        for leg, px in zip(legs, quotes, strict=True):
            fill = px - self.slippage if leg.qty < 0 else px + self.slippage
            fill = max(fill, 0.05)
            cash += -leg.qty * fill
            costs += self._costs(leg, fill, opening=True)
        self.book.position = Position(legs, self.day.date, cash, costs, tag)
        return True

    def close(self, reason: str) -> EodTrade | None:
        pos = self.book.position
        if pos is None:
            return None
        cash = costs = 0.0
        modelled = 0
        for leg in pos.legs:
            px, model = self._px(leg)
            modelled += model
            fill = px + self.slippage if leg.qty < 0 else max(px - self.slippage, 0.0)
            cash += leg.qty * fill
            costs += self._costs(leg, fill, opening=False)
        return self._book_trade(pos, cash, costs, reason, modelled)

    def settle(self) -> EodTrade | None:
        """Expiry: every leg settles at intrinsic value from the index close."""
        pos = self.book.position
        if pos is None:
            return None
        cash = costs = 0.0
        for leg in pos.legs:
            px = self.day.intrinsic(leg.strike, leg.kind)
            cash += leg.qty * px
            units = self.lot * leg.qty
            costs += float(nse_fno.compute_settlement(_KIND[leg.kind], units, _dec(px)).total)
        return self._book_trade(pos, cash, costs, "EXPIRY", 0)

    def _book_trade(self, pos: Position, exit_cash: float, exit_costs: float, reason: str, modelled: int) -> EodTrade:
        points = pos.entry_cash + exit_cash
        costs = pos.entry_costs + exit_costs
        trade = EodTrade(
            pos.entry_date,
            self.day.date,
            " ".join(leg.label() for leg in pos.legs),
            round(pos.entry_cash, 2),
            round(exit_cash, 2),
            round(points, 2),
            round(points * self.lot - costs, 2),
            round(costs, 2),
            reason,
            pos.tag,
            modelled,
        )
        self.book.trades.append(trade)
        self.book.realised += trade.rupees_net
        self.book.modelled_legs += modelled
        self.book.priced_legs += len(pos.legs)
        self.book.position = None
        return trade

    def mark_rupees(self) -> float:
        pos = self.book.position
        if pos is None:
            return self.book.realised
        open_points = pos.entry_cash + sum(leg.qty * self._px(leg)[0] for leg in pos.legs)
        return self.book.realised + open_points * self.lot - pos.entry_costs


class Strategy:
    """Subclasses implement on_day(day, ctx). `params` is shown in reports."""

    label = "strategy"

    def __init__(self, **params) -> None:
        self.params = params

    def on_day(self, day: DayChain, ctx: Ctx) -> None:  # pragma: no cover - interface
        raise NotImplementedError


def run_eod(
    days: Iterable[DayChain], strategies: list[Strategy], *, lot: int = 65, slippage: float = 1.0
) -> list[Book]:
    books = [Book(s.label, s.params) for s in strategies]
    for day in days:
        for strat, book in zip(strategies, books, strict=True):
            ctx = Ctx(day, book, lot, slippage)
            if book.position is not None and day.date >= book.position.expiry:
                ctx.settle()
            strat.on_day(day, ctx)
            book.equity.append((day.date, ctx.mark_rupees()))
    return books
