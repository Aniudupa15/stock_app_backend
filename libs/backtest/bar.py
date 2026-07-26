"""Historical OHLCV bar for the backtester. Kept minimal and standalone so
`libs/backtest` doesn't depend on the data-service; the data-service adapter
maps `historical_prices` rows into these (post monorepo move)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class Bar:
    day: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
