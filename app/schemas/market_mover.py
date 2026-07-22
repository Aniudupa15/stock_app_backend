from decimal import Decimal

from pydantic import BaseModel


class MarketMoverOut(BaseModel):
    symbol: str
    name: str
    last_price: Decimal
    change: Decimal | None
    change_percent: Decimal | None
    volume: int
