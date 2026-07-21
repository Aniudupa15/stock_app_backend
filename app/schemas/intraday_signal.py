from datetime import date
from decimal import Decimal

from pydantic import BaseModel

DISCLAIMER = (
    "This is an automated technical analysis, not investment advice. "
    "Signals are generated from historical price patterns and indicators, "
    "carry no guarantee of future performance, and should not be the sole "
    "basis for a trading decision. Consult a licensed financial advisor."
)


class IntradaySignalOut(BaseModel):
    symbol: str
    as_of: date | None = None
    has_data: bool = False
    signal: str = "HOLD"  # "BUY" | "SELL" | "HOLD"
    confidence: Decimal = Decimal("0")
    entry_price: Decimal | None = None
    target_price: Decimal | None = None
    stop_loss: Decimal | None = None
    risk_reward_ratio: Decimal | None = None
    probability: Decimal | None = None
    reasoning: list[str] = []
    disclaimer: str = DISCLAIMER
