from enum import Enum

from pydantic import BaseModel, Field


class ChatIntent(str, Enum):
    PORTFOLIO_SUMMARY = "portfolio_summary"
    STOCK_QUOTE = "stock_quote"
    INDICATOR_SUMMARY = "indicator_summary"
    WATCHLIST_SUMMARY = "watchlist_summary"
    ALERTS_SUMMARY = "alerts_summary"
    UNKNOWN = "unknown"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)


class ChatResponse(BaseModel):
    intent: ChatIntent
    answer: str
