from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Enum, Index, String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.entities import NewsCategory
from app.models.base import Base, TimestampMixin


class NewsArticleModel(Base, TimestampMixin):
    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    headline: Mapped[str] = mapped_column(String(512), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    url: Mapped[str] = mapped_column(String(1024), unique=True, nullable=False)
    category: Mapped[NewsCategory] = mapped_column(Enum(NewsCategory, name="news_category"), nullable=False)
    related_symbols: Mapped[list[str]] = mapped_column(ARRAY(String(32)), nullable=False, default=list)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (Index("ix_news_articles_published_at", "published_at"),)
