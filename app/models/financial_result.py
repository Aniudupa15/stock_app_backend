import uuid
from datetime import date

from sqlalchemy import BigInteger, Boolean, Date, ForeignKey, Index, Numeric, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class FinancialResultModel(Base, TimestampMixin):
    __tablename__ = "financial_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False
    )
    period_start: Mapped[date] = mapped_column(Date, nullable=False)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)
    consolidated: Mapped[bool] = mapped_column(Boolean, nullable=False)
    revenue: Mapped[float | None] = mapped_column(Numeric(20, 2), nullable=True)
    profit: Mapped[float | None] = mapped_column(Numeric(20, 2), nullable=True)
    eps_basic: Mapped[float | None] = mapped_column(Numeric(10, 4), nullable=True)
    eps_diluted: Mapped[float | None] = mapped_column(Numeric(10, 4), nullable=True)

    __table_args__ = (
        UniqueConstraint("stock_id", "period_end", "consolidated", name="uq_financial_results_stock_period_consolidated"),
        Index("ix_financial_results_stock_id", "stock_id"),
    )
