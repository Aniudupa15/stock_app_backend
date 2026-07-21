import uuid
from datetime import date

from sqlalchemy import Boolean, Date, Enum, Index, Numeric, String, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.entities import InstrumentType
from app.models.base import Base, TimestampMixin


class StockModel(Base, TimestampMixin):
    __tablename__ = "stocks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    symbol: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    isin: Mapped[str | None] = mapped_column(String(12), unique=True, nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    series: Mapped[str | None] = mapped_column(String(8), nullable=True)
    sector: Mapped[str | None] = mapped_column(String(128), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(128), nullable=True)
    instrument_type: Mapped[InstrumentType] = mapped_column(
        Enum(InstrumentType, name="instrument_type"), nullable=False, default=InstrumentType.EQUITY
    )
    listing_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    face_value: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=text("true"))

    __table_args__ = (
        Index("ix_stocks_name_lower", func.lower(name)),
        Index("ix_stocks_is_active", "is_active"),
    )
