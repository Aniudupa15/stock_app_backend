"""NSE equity trading-cost calculator (Zerodha schedule, verified 2026).

Single source of truth for what a trade costs, shared by the paper engine,
live-fill P&L, and the backtester so that a strategy's simulated economics
match a real Zerodha contract note to the paisa (Phase 1 research §5-§6).

All rates live in a frozen `ChargeSchedule` with defaults matching the
verified 2026 Zerodha/NSE cash-equity rates. They are *config*, not
constants baked into logic - statutory rates change (SEBI/exchange revise
them), so a deployment can override the schedule without touching code.

Rate provenance (re-verify before go-live):
  brokerage    delivery ₹0; intraday min(₹20, 0.03% × turnover) per order
  STT          delivery 0.10% both legs; intraday 0.025% SELL leg only
  exchange txn NSE cash ~0.00297% (₹297/cr) - MOST volatile rate, verify
  SEBI         ₹10/crore = 0.0001% of turnover
  stamp duty   delivery 0.015%, intraday 0.003% - BUY leg only
  GST          18% on (brokerage + exchange txn + SEBI)
  DP charge    delivery SELL only: ₹13.5 + 18% GST, per scrip per day (flat)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from libs.charges.models import Charges, Product, Side, round_paise


@dataclass(frozen=True, slots=True)
class ChargeSchedule:
    """Rate card for NSE cash equity. Fractions are of turnover unless noted."""

    # Brokerage
    intraday_brokerage_rate: Decimal = Decimal("0.0003")  # 0.03%
    intraday_brokerage_cap: Decimal = Decimal("20")  # ₹20 per order
    delivery_brokerage_rate: Decimal = Decimal("0")  # free

    # Securities Transaction Tax
    stt_delivery_rate: Decimal = Decimal("0.001")  # 0.10% each leg
    stt_intraday_sell_rate: Decimal = Decimal("0.00025")  # 0.025% sell leg only

    # Exchange transaction charge (NSE cash) - re-verify; most likely to drift
    exchange_txn_rate: Decimal = Decimal("0.0000297")  # ~0.00297%

    # SEBI turnover fee: ₹10 per crore
    sebi_rate: Decimal = Decimal("0.000001")

    # Stamp duty (BUY leg only)
    stamp_delivery_rate: Decimal = Decimal("0.00015")  # 0.015%
    stamp_intraday_rate: Decimal = Decimal("0.00003")  # 0.003%

    # GST on (brokerage + exchange txn + SEBI)
    gst_rate: Decimal = Decimal("0.18")

    # Depository (DP) charge - delivery sell only, flat per scrip per day
    dp_charge: Decimal = Decimal("13.5")


DEFAULT_NSE_EQUITY_SCHEDULE = ChargeSchedule()


def compute(
    side: Side,
    product: Product,
    quantity: int,
    price: Decimal,
    *,
    schedule: ChargeSchedule = DEFAULT_NSE_EQUITY_SCHEDULE,
) -> Charges:
    """Charges for one executed equity order (a single leg).

    `quantity` is share count, `price` the fill price per share. Returns an
    itemised, paise-rounded `Charges`. Round-trip cost is obtained by summing
    the BUY and SELL results - this function never assumes a round trip.
    """
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if price <= 0:
        raise ValueError("price must be positive")

    turnover = Decimal(quantity) * price

    # Brokerage
    if product is Product.CNC:
        brokerage = schedule.delivery_brokerage_rate * turnover
    else:  # MIS
        brokerage = min(schedule.intraday_brokerage_cap, schedule.intraday_brokerage_rate * turnover)

    # STT
    if product is Product.CNC:
        stt = schedule.stt_delivery_rate * turnover  # both legs
    else:
        stt = schedule.stt_intraday_sell_rate * turnover if side is Side.SELL else Decimal("0")

    # Exchange transaction charge (both legs)
    exchange_txn = schedule.exchange_txn_rate * turnover

    # SEBI turnover fee (both legs)
    sebi = schedule.sebi_rate * turnover

    # Stamp duty (buy leg only)
    if side is Side.BUY:
        stamp_rate = schedule.stamp_delivery_rate if product is Product.CNC else schedule.stamp_intraday_rate
        stamp_duty = stamp_rate * turnover
    else:
        stamp_duty = Decimal("0")

    # GST on brokerage + exchange txn + SEBI
    gst = schedule.gst_rate * (brokerage + exchange_txn + sebi)

    # DP charge - delivery sell only (flat per scrip per day, incl. its own GST)
    if product is Product.CNC and side is Side.SELL:
        dp = schedule.dp_charge * (Decimal("1") + schedule.gst_rate)
    else:
        dp = Decimal("0")

    return Charges(
        brokerage=round_paise(brokerage),
        stt=round_paise(stt),
        exchange_txn=round_paise(exchange_txn),
        sebi=round_paise(sebi),
        stamp_duty=round_paise(stamp_duty),
        gst=round_paise(gst),
        dp=round_paise(dp),
    )
