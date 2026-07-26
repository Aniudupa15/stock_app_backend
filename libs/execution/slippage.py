"""Deterministic slippage model for paper/backtest fills.

Deterministic on purpose: no randomness (so backtests are reproducible and
tests are exact), yet realistic - slippage widens with order size relative to
the day's volume and with illiquidity. The goal (Phase 1 §5) is a *slightly
pessimistic* proxy of live, never a rosier one.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from libs.trading_domain.enums import Side


@dataclass(frozen=True, slots=True)
class SlippageModel:
    base_bps: Decimal = Decimal("3")  # baseline for a liquid name
    impact_coeff_bps: Decimal = Decimal("500")  # * participation (qty/day_volume)
    illiquid_threshold: int = 50_000  # day_volume below this = illiquid
    illiquid_penalty_bps: Decimal = Decimal("15")
    max_bps: Decimal = Decimal("200")  # cap (2%)

    def bps(self, quantity: int, day_volume: int | None) -> Decimal:
        total = self.base_bps
        if day_volume and day_volume > 0:
            participation = Decimal(quantity) / Decimal(day_volume)
            total += participation * self.impact_coeff_bps
            if day_volume < self.illiquid_threshold:
                total += self.illiquid_penalty_bps
        else:
            # No volume info -> assume worst (illiquid) rather than best.
            total += self.illiquid_penalty_bps
        return min(total, self.max_bps)

    def adjust(self, side: Side, reference_price: Decimal, quantity: int, day_volume: int | None) -> Decimal:
        """Apply slippage against the trader: BUY fills higher, SELL lower."""
        fraction = self.bps(quantity, day_volume) / Decimal("10000")
        if side is Side.BUY:
            return reference_price * (Decimal("1") + fraction)
        return reference_price * (Decimal("1") - fraction)
