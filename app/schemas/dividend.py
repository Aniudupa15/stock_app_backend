from datetime import date
from decimal import Decimal

from pydantic import BaseModel


class DividendRecommendationOut(BaseModel):
    symbol: str
    name: str
    dividend_yield: Decimal
    dividend_amount: Decimal
    ex_dividend_date: date
    buy_before_date: date
    recommendation: str
    risk_level: str
    confidence: int
