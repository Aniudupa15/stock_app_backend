import uuid
from datetime import date

from sqlalchemy import BigInteger, Date, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class LongTermSignalSnapshotModel(Base, TimestampMixin):
    __tablename__ = "long_term_signal_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("stocks.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    as_of: Mapped[date] = mapped_column(Date, nullable=False)
    signal: Mapped[str] = mapped_column(String(10), nullable=False)
    confidence: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)
    growth_potential: Mapped[str] = mapped_column(String(20), nullable=False)
    investment_tenure: Mapped[str] = mapped_column(String(20), nullable=False)
    reasoning: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
