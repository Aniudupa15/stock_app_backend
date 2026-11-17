"""Value objects for the trading-cost calculator.

Kept deliberately standalone (its own `Side`/`Product` enums rather than
importing from the data-service domain) so `libs/charges` stays a pure,
dependency-free package that the paper engine, live P&L, and backtester can
all share without pulling in SQLAlchemy or the NSE provider. When
`libs/trading_domain` lands (Phase 4b) it can re-export these rather than
duplicate them.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum

_PAISE = Decimal("0.01")


def round_paise(value: Decimal) -> Decimal:
    """Round a rupee amount to the nearest paisa (2 dp), half-up.

    Every statutory charge is quantised the same way so paper-mode P&L lines
    up to the paisa with a live broker contract note.
    """
    return value.quantize(_PAISE, rounding=ROUND_HALF_UP)


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Product(str, Enum):
    """Zerodha product codes.

    MIS  = intraday (auto-squared off same day) - equity or F&O.
    CNC  = equity delivery (T+1 settled). Not valid for F&O.
    NRML = F&O carry-forward (held overnight up to expiry). Not valid for
           cash equity; its charges come from `libs.charges.nse_fno`.
    """

    MIS = "MIS"
    CNC = "CNC"
    NRML = "NRML"


@dataclass(frozen=True, slots=True)
class Charges:
    """Itemised statutory + broker charges for a single executed order (one leg).

    Round-trip cost = charges(BUY) + charges(SELL); nothing here nets the two,
    because a fill is a single-sided event and that is exactly what the paper
    engine and live fill-handler each produce.
    """

    brokerage: Decimal
    stt: Decimal
    exchange_txn: Decimal
    sebi: Decimal
    stamp_duty: Decimal
    gst: Decimal
    dp: Decimal

    @property
    def total(self) -> Decimal:
        return round_paise(
            self.brokerage + self.stt + self.exchange_txn + self.sebi + self.stamp_duty + self.gst + self.dp
        )

    def as_dict(self) -> dict[str, str]:
        """JSON-friendly (string Decimals) - what gets persisted in `fills.charges`."""
        return {
            "brokerage": str(self.brokerage),
            "stt": str(self.stt),
            "exchange_txn": str(self.exchange_txn),
            "sebi": str(self.sebi),
            "stamp_duty": str(self.stamp_duty),
            "gst": str(self.gst),
            "dp": str(self.dp),
            "total": str(self.total),
        }
