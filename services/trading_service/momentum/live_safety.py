"""Pre-trade safety gates for the LIVE Zerodha rebalance. Pure - no I/O.

Two layers, both run every time before any money moves:

  Data gates (before emailing the approval link) - is the ranking built on
  fresh, complete, sane prices?
    - freshness: latest bar == previous NSE trading day (fails SAFE: an
      unlisted holiday blocks rather than trading on stale data)
    - coverage: the latest day loaded fully (not a half-written Bhavcopy)
    - pick sanity: drop picks whose history shows an absurd 30-day return or
      a single-day jump - almost always an unadjusted split/bonus or bad bar,
      not real momentum
    - enough clean picks left to fill the portfolio

  Order gates (after login, before placing) - is what we're about to send sane?
    - correct Zerodha account logged in
    - live price vs DB close deviation (split/bad data caught at the last moment)
    - every buy within its slot, total bot capital within the cap, order count bounded
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from libs.trading_calendar.calendar import TradingCalendar
from services.trading_service.momentum.live_plan import LiveOrder


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str

    def line(self) -> str:
        return f"  [{'PASS' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


def check_freshness(latest: date | None, today: date, holidays: set[date]) -> Check:
    expected = TradingCalendar(holidays).previous_trading_day(today)
    if latest is None:
        return Check("data freshness", False, "no price data at all")
    if latest < expected:
        return Check(
            "data freshness",
            False,
            f"latest close {latest}, expected {expected}. If {expected} was an NSE holiday, "
            f"add it to NSE_HOLIDAYS in .env",
        )
    return Check("data freshness", True, f"latest close {latest}")


def check_coverage(latest_count: int, prev_count: int, *, min_count: int = 1500, min_ratio: float = 0.9) -> Check:
    ok = latest_count >= min_count and (prev_count == 0 or latest_count >= prev_count * min_ratio)
    return Check("data coverage", ok, f"{latest_count} stocks priced on latest day (previous day {prev_count})")


def pick_problem(
    closes: list[float], *, lookback: int = 30, max_return_pct: float = 200, max_daily_pct: float = 35
) -> str | None:
    """Why a candidate's price series is untrustworthy, or None if it looks clean."""
    window = closes[-(lookback + 1) :]
    if len(window) < lookback + 1 or min(window) <= 0:
        return "incomplete price history"
    ret = (window[-1] / window[0] - 1) * 100
    if ret > max_return_pct:
        return f"{ret:.0f}% in {lookback}d looks like an unadjusted split/bonus or bad data"
    for a, b in zip(window, window[1:], strict=False):
        move = abs(b / a - 1) * 100
        if move > max_daily_pct:
            return f"single-day move of {move:.0f}% (likely corporate action / bad bar)"
    return None


def filter_picks(
    candidates: list[str], series: dict[str, list[float]], top: int, **limits
) -> tuple[list[str], list[str], Check]:
    """Walk the ranked candidates, drop unsafe ones, keep the first `top` clean."""
    picks, excluded = [], []
    for symbol in candidates:
        problem = pick_problem(series.get(symbol, []), **limits)
        if problem:
            excluded.append(f"{symbol}: {problem}")
        else:
            picks.append(symbol)
        if len(picks) == top:
            break
    ok = len(picks) == top
    return picks, excluded, Check("pick sanity", ok, f"{len(picks)}/{top} clean picks, {len(excluded)} excluded")


def check_account(expected_user_id: str | None, actual_user_id: str) -> Check:
    if not expected_user_id:
        return Check("broker account", True, f"logged in as {actual_user_id} (set KITE_USER_ID to enforce)")
    ok = expected_user_id.strip().upper() == actual_user_id.strip().upper()
    return Check("broker account", ok, f"expected {expected_user_id}, logged in as {actual_user_id}")


def price_deviations(
    live: dict[str, Decimal], db_close: dict[str, Decimal], *, max_dev_pct: Decimal = Decimal(20)
) -> dict[str, str]:
    """Symbols whose live price is far from the ranked close - skip them this run."""
    out = {}
    for symbol, ltp in live.items():
        close = db_close.get(symbol)
        if not close or close <= 0:
            continue
        dev = abs(ltp / close - 1) * 100
        if dev > max_dev_pct:
            out[symbol] = f"live Rs{ltp} vs close Rs{close} ({dev:.0f}% apart)"
    return out


def check_buy_orders(
    orders: list[LiveOrder],
    *,
    kept_value: Decimal,
    per_stock: Decimal,
    capital_cap: Decimal,
    top: int,
    max_slot_overrun: Decimal = Decimal("1.5"),
) -> Check:
    """Last guard against a sizing bug sending an oversized order. A single
    share may overrun its slot by up to `max_slot_overrun` (see plan_buys)."""
    tol = Decimal("1.02")
    if len(orders) > top:
        return Check("order limits", False, f"{len(orders)} buys > top-{top}")
    for o in orders:
        value = o.qty * o.ref_price
        limit = per_stock * (max_slot_overrun if o.qty == 1 else tol)
        if o.qty <= 0 or value > limit:
            return Check("order limits", False, f"{o.symbol} x{o.qty} = Rs{value:,.0f} exceeds slot Rs{per_stock:,.0f}")
    total = kept_value + sum((o.qty * o.ref_price for o in orders), Decimal(0))
    if total > capital_cap * tol:
        return Check("order limits", False, f"bot capital after buys Rs{total:,.0f} > cap Rs{capital_cap:,.0f}")
    return Check(
        "order limits", True, f"{len(orders)} buys, bot capital after Rs{total:,.0f} <= cap Rs{capital_cap:,.0f}"
    )
