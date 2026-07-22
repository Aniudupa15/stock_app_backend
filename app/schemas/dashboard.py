from pydantic import BaseModel

from app.schemas.market_mover import MarketMoverOut
from app.schemas.market_status import IndexQuoteOut, MarketStatusOut
from app.schemas.news import NewsArticleOut


class DashboardOut(BaseModel):
    market_status: list[MarketStatusOut]
    indices: list[IndexQuoteOut]
    gainers: list[MarketMoverOut]
    losers: list[MarketMoverOut]
    most_active: list[MarketMoverOut]
    fifty_two_week_high: list[MarketMoverOut]
    fifty_two_week_low: list[MarketMoverOut]
    latest_news: list[NewsArticleOut]
    # Explicit, visible limitations rather than silently blank/missing fields -
    # see Phase 4 plan's "Scope reductions" section.
    notes: list[str]
