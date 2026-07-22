from datetime import date
from decimal import Decimal

from pydantic import BaseModel


class OhlcvBarOut(BaseModel):
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


class HistoryOut(BaseModel):
    symbol: str
    range: str
    bars: list[OhlcvBarOut]
