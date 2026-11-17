"""Paper-mode F&O margin approximation.

Real brokers block SPAN + exposure margin computed by the exchange's risk
arrays. Paper mode can't run SPAN, so it uses a flat-percentage proxy that is
deliberately on the conservative side of typical NSE requirements:

  * futures (long or short): notional × pct   (index ~12%, stock ~20%)
  * long options:             nothing blocked - the premium is paid in full
  * short options:            strike × qty × pct (same pct as futures), on top
                              of which the premium received sits in cash

Treat the resulting numbers as "will this roughly fit in my account", not as
the exact figure Zerodha's order-margin calculator would show.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from libs.derivatives.contracts import FnoContract, InstrumentKind


@dataclass(frozen=True, slots=True)
class FnoMarginModel:
    index_pct: Decimal = Decimal("0.12")
    stock_pct: Decimal = Decimal("0.20")

    def pct(self, contract: FnoContract) -> Decimal:
        return self.index_pct if contract.is_index else self.stock_pct

    def per_unit(self, contract: FnoContract, net_qty_sign: int, price: Decimal) -> Decimal:
        """Margin blocked per unit for a position of the given direction
        (+1 long / -1 short) at `price` (futures/option LTP)."""
        if contract.kind is InstrumentKind.FUT:
            return price * self.pct(contract)
        if net_qty_sign > 0:
            return Decimal("0")  # long option: premium already paid
        assert contract.strike is not None
        return contract.strike * self.pct(contract)

    def blocked(self, contract: FnoContract, net_qty: int, price: Decimal) -> Decimal:
        if net_qty == 0:
            return Decimal("0")
        sign = 1 if net_qty > 0 else -1
        return self.per_unit(contract, sign, price) * abs(net_qty)

    def entry_cost_per_unit(self, contract: FnoContract, side_sign: int, price: Decimal) -> Decimal:
        """Funds an opening order needs per unit: margin, plus the premium for
        an option BUY. Used by the risk gate to size positions."""
        if contract.kind.is_option and side_sign > 0:
            return price
        return self.per_unit(contract, side_sign, price)
