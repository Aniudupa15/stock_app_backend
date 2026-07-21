import uuid
from datetime import date

from sqlalchemy import BigInteger, Date, ForeignKey, Index, Numeric, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class CorporateActionModel(Base, TimestampMixin):
    __tablename__ = "corporate_actions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False
    )
    purpose: Mapped[str] = mapped_column(String(512), nullable=False)
    face_value: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    ex_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    record_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    book_closure_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    book_closure_end: Mapped[date | None] = mapped_column(Date, nullable=True)

    __table_args__ = (
        UniqueConstraint("stock_id", "purpose", "ex_date", name="uq_corporate_actions_stock_purpose_exdate"),
        Index("ix_corporate_actions_stock_id", "stock_id"),
    )
