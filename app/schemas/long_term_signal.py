from pydantic import BaseModel

from app.schemas.intraday_signal import DISCLAIMER


class LongTermSignalOut(BaseModel):
    symbol: str
    has_data: bool = False
    signal: str = "HOLD"  # "BUY" | "HOLD" | "AVOID"
    confidence: int = 0
    investment_horizon: str = "Long-term (1-3+ years)"
    risk_level: str = "Unknown"
    growth_potential: str = "Unknown"
    strengths: list[str] = []
    weaknesses: list[str] = []
    opportunities: list[str] = []
    risks: list[str] = []
    reasoning: list[str] = []
    disclaimer: str = DISCLAIMER
