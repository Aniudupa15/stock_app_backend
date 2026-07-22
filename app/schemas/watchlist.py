import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class WatchlistCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)


class WatchlistAddSymbol(BaseModel):
    symbol: str = Field(..., min_length=1)


class WatchlistOut(BaseModel):
    id: uuid.UUID
    name: str
    created_at: datetime
    item_count: int


class WatchlistItemOut(BaseModel):
    symbol: str
    name: str
    added_at: datetime
    last_price: Decimal | None
    change: Decimal | None
    change_percent: Decimal | None


class WatchlistDetailOut(BaseModel):
    id: uuid.UUID
    name: str
    created_at: datetime
    items: list[WatchlistItemOut]
