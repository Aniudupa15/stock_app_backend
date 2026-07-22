"""Weighted-average-cost holdings, derived from a transaction log. A SELL
reduces cost basis proportionally to the average cost at the time of sale
(not FIFO/LIFO lot matching) - the standard "average cost" method retail
brokerages use when they don't track individual lots.
"""

from dataclasses import dataclass
from decimal import Decimal

from app.domain.entities import PortfolioTransaction, TransactionType


@dataclass(frozen=True, slots=True)
class Holding:
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    cost_basis: Decimal


def compute_holdings(transactions: list[PortfolioTransaction]) -> list[Holding]:
    accumulators: dict[str, tuple[Decimal, Decimal]] = {}

    for t in sorted(transactions, key=lambda t: t.transaction_date):
        qty, cost = accumulators.get(t.symbol, (Decimal(0), Decimal(0)))
        if t.transaction_type == TransactionType.BUY:
            qty += t.quantity
            cost += t.quantity * t.price
        else:
            if qty > 0:
                avg_price = cost / qty
                cost -= t.quantity * avg_price
            qty -= t.quantity
        accumulators[t.symbol] = (qty, cost)

    holdings = []
    for symbol, (qty, cost) in accumulators.items():
        if qty <= 0:
            continue
        holdings.append(Holding(symbol=symbol, quantity=qty, avg_price=cost / qty, cost_basis=cost))
    return holdings
