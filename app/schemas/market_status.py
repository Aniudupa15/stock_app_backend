from decimal import Decimal

from pydantic import BaseModel


class MarketStatusOut(BaseModel):
    market: str
    status: str
    as_of: str


class IndexQuoteOut(BaseModel):
    index_name: str
    last_price: Decimal
    change: Decimal
    change_percent: Decimal
