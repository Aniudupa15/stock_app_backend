from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field


class ScreenerRequest(BaseModel):
    rsi_below: Decimal | None = None
    rsi_above: Decimal | None = None
    price_min: Decimal | None = None
    price_max: Decimal | None = None
    above_sma_50: bool | None = None
    min_volume: int | None = None
    limit: int = Field(50, ge=1, le=200)


class ScreenerResultOut(BaseModel):
    symbol: str
    name: str
    as_of: date
    close: Decimal
    volume: int
    rsi_14: Decimal | None
    sma_50: Decimal | None
    sma_200: Decimal | None
