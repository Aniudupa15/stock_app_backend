from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel

from app.domain.entities import InstrumentType


class StockSearchResult(BaseModel):
    symbol: str
    name: str
    isin: str | None
    series: str | None
    instrument_type: InstrumentType


class QuoteOut(BaseModel):
    last_price: Decimal
    change: Decimal
    change_percent: Decimal
    open: Decimal
    high: Decimal
    low: Decimal
    previous_close: Decimal
    volume: int
    as_of: datetime


class StockDetail(BaseModel):
    symbol: str
    isin: str | None
    name: str
    series: str | None
    sector: str | None
    industry: str | None
    instrument_type: InstrumentType
    listing_date: date | None
    face_value: Decimal | None
    quote: QuoteOut | None
    quote_unavailable_reason: str | None = None
