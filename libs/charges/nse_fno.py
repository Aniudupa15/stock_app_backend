"""NSE F&O trading-cost calculator (Zerodha schedule).

Same contract as `nse_equity.compute`: charges for ONE executed leg, itemised
and paise-rounded, so paper F&O P&L lines up with a contract note. Turnover
is quantity (units, i.e. lots x lot size) x price - for options that price is
the PREMIUM, which is what NSE levies options charges on.

Rate provenance (all config - re-verify before go-live, statutory rates move):
  brokerage    futures min(₹20, 0.03% × turnover); options flat ₹20 per order
  STT          futures 0.05% SELL; options 0.15% of premium SELL
               (Union Budget 2026 rates, effective 1 Apr 2026; were 0.02% / 0.1%)
               exercised ITM options 0.15% of intrinsic value (BUY side, at expiry)
  exchange txn NSE futures 0.00173%; options 0.03503% of premium
  SEBI         ₹10/crore
  stamp duty   futures 0.002%, options 0.003% - BUY leg only
  GST          18% on (brokerage + exchange txn + SEBI)
  DP           none (F&O has no demat delivery leg)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from libs.charges.models import Charges, Side, round_paise
from libs.derivatives.contracts import InstrumentKind


@dataclass(frozen=True, slots=True)
class FnoChargeSchedule:
    futures_brokerage_rate: Decimal = Decimal("0.0003")
    brokerage_cap: Decimal = Decimal("20")  # per executed order (futures cap, options flat)

    stt_futures_sell_rate: Decimal = Decimal("0.0005")  # 0.05%
    stt_options_sell_rate: Decimal = Decimal("0.0015")  # 0.15% of premium
    stt_exercise_rate: Decimal = Decimal("0.0015")  # 0.15% of intrinsic on exercise

    exchange_txn_futures_rate: Decimal = Decimal("0.0000173")  # 0.00173%
    exchange_txn_options_rate: Decimal = Decimal("0.0003503")  # 0.03503% of premium

    sebi_rate: Decimal = Decimal("0.000001")  # ₹10/crore

    stamp_futures_rate: Decimal = Decimal("0.00002")  # 0.002%
    stamp_options_rate: Decimal = Decimal("0.00003")  # 0.003%

    gst_rate: Decimal = Decimal("0.18")


DEFAULT_NSE_FNO_SCHEDULE = FnoChargeSchedule()

_ZERO = Decimal("0")


def compute(
    side: Side,
    kind: InstrumentKind,
    quantity: int,
    price: Decimal,
    *,
    schedule: FnoChargeSchedule = DEFAULT_NSE_FNO_SCHEDULE,
) -> Charges:
    """Charges for one executed F&O order. `quantity` is units (not lots)."""
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if price < 0:
        raise ValueError("price must be non-negative")

    turnover = Decimal(quantity) * price

    if kind is InstrumentKind.FUT:
        brokerage = min(schedule.brokerage_cap, schedule.futures_brokerage_rate * turnover)
        stt = schedule.stt_futures_sell_rate * turnover if side is Side.SELL else _ZERO
        exchange_txn = schedule.exchange_txn_futures_rate * turnover
        stamp = schedule.stamp_futures_rate * turnover if side is Side.BUY else _ZERO
    else:
        # Zerodha charges a flat ₹20 per executed option order (never more
        # than the premium itself on a near-worthless contract).
        brokerage = min(schedule.brokerage_cap, turnover) if turnover > 0 else _ZERO
        stt = schedule.stt_options_sell_rate * turnover if side is Side.SELL else _ZERO
        exchange_txn = schedule.exchange_txn_options_rate * turnover
        stamp = schedule.stamp_options_rate * turnover if side is Side.BUY else _ZERO

    sebi = schedule.sebi_rate * turnover
    gst = schedule.gst_rate * (brokerage + exchange_txn + sebi)

    return Charges(
        brokerage=round_paise(brokerage),
        stt=round_paise(stt),
        exchange_txn=round_paise(exchange_txn),
        sebi=round_paise(sebi),
        stamp_duty=round_paise(stamp),
        gst=round_paise(gst),
        dp=_ZERO,
    )


def compute_settlement(
    kind: InstrumentKind,
    net_qty: int,
    settlement_price: Decimal,
    *,
    schedule: FnoChargeSchedule = DEFAULT_NSE_FNO_SCHEDULE,
) -> Charges:
    """Charges when a position is settled by the exchange at expiry (no order,
    so no brokerage). Only a LONG in-the-money option attracts exercise STT;
    futures and short options settle without STT."""
    stt = _ZERO
    if kind.is_option and net_qty > 0 and settlement_price > 0:
        stt = schedule.stt_exercise_rate * Decimal(net_qty) * settlement_price
    return Charges(
        brokerage=_ZERO,
        stt=round_paise(stt),
        exchange_txn=_ZERO,
        sebi=_ZERO,
        stamp_duty=_ZERO,
        gst=_ZERO,
        dp=_ZERO,
    )
