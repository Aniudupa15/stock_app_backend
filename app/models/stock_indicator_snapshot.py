import uuid
from datetime import date

from sqlalchemy import BigInteger, Date, ForeignKey, Numeric
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class StockIndicatorSnapshotModel(Base, TimestampMixin):
    __tablename__ = "stock_indicator_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("stocks.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    as_of: Mapped[date] = mapped_column(Date, nullable=False)
    close: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    rsi_14: Mapped[float | None] = mapped_column(Numeric(10, 4), nullable=True)
    sma_50: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    sma_200: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
