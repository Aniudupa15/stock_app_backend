import uuid
from datetime import date

from sqlalchemy import BigInteger, Date, ForeignKey, Numeric, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class IntradaySignalSnapshotModel(Base, TimestampMixin):
    __tablename__ = "intraday_signal_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("stocks.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    as_of: Mapped[date] = mapped_column(Date, nullable=False)
    signal: Mapped[str] = mapped_column(String(10), nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    entry_price: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    target_price: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    stop_loss: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    reasoning: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
