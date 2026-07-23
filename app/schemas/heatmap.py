from decimal import Decimal

from pydantic import BaseModel


class HeatmapTileOut(BaseModel):
    symbol: str
    name: str
    last_price: Decimal
    change_percent: Decimal | None
    volume: int
    bucket: str  # STRONG_GAIN / GAIN / FLAT / LOSS / STRONG_LOSS / UNKNOWN


class HeatmapOut(BaseModel):
    tiles: list[HeatmapTileOut]
    notes: list[str]
