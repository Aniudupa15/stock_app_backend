import uuid
from datetime import date, datetime

from sqlalchemy import BigInteger, Date, DateTime, ForeignKey, Index, Numeric, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class HistoricalPriceModel(Base):
    __tablename__ = "historical_prices"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False
    )
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    high: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    low: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("stock_id", "trade_date", name="uq_historical_prices_stock_date"),
        Index("ix_hist_stock_date", "stock_id", text("trade_date DESC")),
    )
