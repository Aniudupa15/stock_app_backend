"""Enumerations for the trading domain.

`Side` and `Product` are re-exported from `libs.charges` (the lower-level
leaf package) rather than redefined, so there is exactly one canonical
`Side`/`Product` identity across charges, the paper engine, and the OMS - no
mapping, no drift. `Product` is MIS/CNC (equity) only; NRML/derivatives are
deferred to a future phase.
"""

from __future__ import annotations

from enum import Enum

from libs.charges.models import Product, Side  # re-export canonical enums

__all__ = [
    "Side",
    "Product",
    "OrderType",
    "Validity",
    "Mode",
    "Venue",
    "OrderState",
    "Leg",
    "ExitReason",
]


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"  # stop-loss limit (trigger + price)
    SL_M = "SL_M"  # stop-loss market (trigger only)


class Validity(str, Enum):
    DAY = "DAY"
    IOC = "IOC"
    TTL = "TTL"


class Mode(str, Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"


class Venue(str, Enum):
    PAPER = "PAPER"
    ZERODHA = "ZERODHA"
    BACKTEST = "BACKTEST"


class OrderState(str, Enum):
    """Normalised lifecycle (broker-specific statuses map into these)."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    OPEN = "OPEN"  # resting / armed (LIMIT not marketable, SL trigger pending)
    PARTIAL = "PARTIAL"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Leg(str, Enum):
    ENTRY = "ENTRY"
    SL = "SL"
    TARGET = "TARGET"


class ExitReason(str, Enum):
    TARGET = "TARGET"
    STOP_LOSS = "STOP_LOSS"
    SQUARE_OFF = "SQUARE_OFF"
    MANUAL = "MANUAL"
    SIGNAL = "SIGNAL"
