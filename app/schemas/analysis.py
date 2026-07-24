from datetime import date
from decimal import Decimal

from pydantic import BaseModel


class IntradayRecommendationOut(BaseModel):
    symbol: str
    name: str
    as_of: date
    signal: str
    confidence: Decimal
    entry_price: Decimal | None
    target_price: Decimal | None
    stop_loss: Decimal | None
    reasoning: list[str]


class LongTermRecommendationOut(BaseModel):
    symbol: str
    name: str
    as_of: date
    signal: str
    confidence: int
    risk_level: str
    growth_potential: str
    investment_tenure: str
    reasoning: list[str]
