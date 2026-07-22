import uuid
from datetime import date, datetime

from sqlalchemy import BigInteger, Date, DateTime, Enum, ForeignKey, Index, Numeric, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.entities import TransactionType
from app.models.base import Base


class PortfolioTransactionModel(Base):
    __tablename__ = "portfolio_transactions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False
    )
    stock_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False
    )
    transaction_type: Mapped[TransactionType] = mapped_column(
        Enum(TransactionType, name="transaction_type"), nullable=False
    )
    quantity: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    price: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    transaction_date: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_portfolio_transactions_portfolio_id", "portfolio_id"),)
