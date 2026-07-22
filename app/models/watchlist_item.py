import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class WatchlistItemModel(Base):
    __tablename__ = "watchlist_items"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    watchlist_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("watchlists.id", ondelete="CASCADE"), nullable=False
    )
    stock_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False
    )
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("watchlist_id", "stock_id", name="uq_watchlist_items_watchlist_stock"),
        Index("ix_watchlist_items_watchlist_id", "watchlist_id"),
    )
