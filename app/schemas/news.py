from datetime import datetime

from pydantic import BaseModel

from app.domain.entities import NewsCategory


class NewsArticleOut(BaseModel):
    headline: str
    summary: str | None
    source: str
    url: str
    category: NewsCategory
    related_symbols: list[str]
    published_at: datetime
