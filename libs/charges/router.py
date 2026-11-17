"""Pick the right charge schedule for an order: NSE cash equity vs F&O.

The exchange decides, not the symbol - an equity symbol can end in "CE"."""

from __future__ import annotations

from decimal import Decimal

from libs.charges import nse_equity, nse_fno
from libs.charges.models import Charges, Product, Side
from libs.derivatives.contracts import FNO_EXCHANGES, kind_from_fno_symbol


def charges_for(exchange: str, symbol: str, side: Side, product: Product, quantity: int, price: Decimal) -> Charges:
    if exchange in FNO_EXCHANGES:
        return nse_fno.compute(side, kind_from_fno_symbol(symbol), quantity, price)
    return nse_equity.compute(side, product, quantity, price)
