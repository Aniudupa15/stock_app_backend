from pydantic import BaseModel

from app.schemas.fundamentals import FundamentalsOut
from app.schemas.indicators import IndicatorsOut
from app.schemas.stock import StockDetail


class ComparisonEntryOut(BaseModel):
    detail: StockDetail
    indicators: IndicatorsOut
    fundamentals: FundamentalsOut


class ComparisonOut(BaseModel):
    entries: list[ComparisonEntryOut]
