"""Pure planning logic for the LIVE (Zerodha) monthly momentum rebalance.

No broker, DB or network here - scripts/zerodha_rebalance.py does the I/O and
calls these. Everything works off the bot's own ledger, so holdings the bot
did not buy are never sold, and total bot capital never exceeds the cap.

Unlike the paper rebalance (sell-all -> buy-all), this only trades the diff:
stocks that stay in the top-N are kept, saving two legs of charges each.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal


@dataclass(frozen=True)
class LiveOrder:
    symbol: str
    side: str  # "BUY" | "SELL"
    qty: int
    ref_price: Decimal


@dataclass
class BuyPlan:
    orders: list[LiveOrder]
    budget: Decimal
    per_stock: Decimal
    skipped: list[str] = field(default_factory=list)  # too expensive for one slot


def round_to_tick(price: Decimal, tick: Decimal, *, up: bool) -> Decimal:
    mode = ROUND_CEILING if up else ROUND_FLOOR
    return ((price / tick).to_integral_value(rounding=mode) * tick).quantize(tick)


def reconcile_ledger(ledger: dict[str, int], broker_qty: dict[str, int]) -> dict[str, int]:
    """The bot owns min(what it recorded, what the broker actually holds).

    Covers a GTT stop-loss having fired (broker has fewer/none) and never lets
    the bot claim shares you bought yourself (broker has more)."""
    out = {}
    for symbol, qty in ledger.items():
        held = min(qty, broker_qty.get(symbol, 0))
        if held > 0:
            out[symbol] = held
    return out


def plan_sells(picks: list[str], ledger: dict[str, int], prices: dict[str, Decimal]) -> list[LiveOrder]:
    keep = set(picks)
    return [
        LiveOrder(symbol, "SELL", qty, prices.get(symbol, Decimal(0)))
        for symbol, qty in sorted(ledger.items())
        if symbol not in keep and qty > 0
    ]


def plan_buys(
    picks: list[str],
    ledger_after_sells: dict[str, int],
    prices: dict[str, Decimal],
    *,
    free_cash: Decimal,
    capital_cap: Decimal,
    top: int,
    cash_buffer: Decimal = Decimal("0.98"),
    max_slot_overrun: Decimal = Decimal("1.5"),
) -> BuyPlan:
    """Equal-weight slots of min(cap, bot equity + free cash) / top for each new pick.

    Kept positions are not resized (avoids churn). Buys are further limited by
    real free cash (x buffer for charges and price drift since the reference).

    A pick whose single share costs more than its slot still gets ONE share if
    it is within `max_slot_overrun` x the slot and the leftover cash covers it -
    otherwise small accounts would skip every high-priced pick. Affordable picks
    are sized first so an overrun never starves them."""
    kept_value = sum(
        (Decimal(qty) * prices.get(sym, Decimal(0)) for sym, qty in ledger_after_sells.items()),
        Decimal(0),
    )
    budget = min(capital_cap, kept_value + free_cash)
    per_stock = budget / top if top > 0 else Decimal(0)

    new = [s for s in picks[:top] if s not in ledger_after_sells and prices.get(s, Decimal(0)) > 0]
    spendable = free_cash * cash_buffer
    # Never spend more cash than the budget leaves room for, or than we actually have.
    room = max(Decimal(0), budget - kept_value)
    alloc = min(per_stock, spendable / len(new), room / len(new)) if new else Decimal(0)

    cash_left = min(spendable, room)
    sized: dict[str, int] = {}
    for symbol in new:
        qty = int((alloc / prices[symbol]).to_integral_value(rounding=ROUND_FLOOR))
        if qty > 0:
            sized[symbol] = qty
            cash_left -= qty * prices[symbol]
    skipped = []
    for symbol in new:
        price = prices[symbol]
        if symbol in sized:
            continue
        if price <= alloc * max_slot_overrun and price <= cash_left:
            sized[symbol] = 1
            cash_left -= price
        else:
            skipped.append(symbol)
    orders = [LiveOrder(s, "BUY", sized[s], prices[s]) for s in new if s in sized]
    return BuyPlan(orders=orders, budget=budget, per_stock=per_stock, skipped=skipped)


def stop_loss_prices(avg_price: Decimal, stop_pct: Decimal, tick: Decimal) -> tuple[Decimal, Decimal]:
    """(trigger, limit) for a GTT stop-loss sell. The limit sits 3% under the
    trigger so a gap-down still fills instead of resting unfilled."""
    trigger = round_to_tick(avg_price * (1 - stop_pct / 100), tick, up=False)
    limit = round_to_tick(trigger * Decimal("0.97"), tick, up=False)
    return trigger, max(limit, tick)
