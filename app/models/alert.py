import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Index, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.entities import AlertStatus, AlertType
from app.models.base import Base


class AlertModel(Base):
    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    # Nullable to leave room for a future portfolio-wide alert (stock_id IS
    # NULL); nothing in this phase creates or evaluates one yet.
    stock_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("stocks.id", ondelete="CASCADE"), nullable=True
    )
    alert_type: Mapped[AlertType] = mapped_column(Enum(AlertType, name="alert_type"), nullable=False)
    condition: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[AlertStatus] = mapped_column(
        Enum(AlertStatus, name="alert_status"), nullable=False, default=AlertStatus.ACTIVE, server_default="ACTIVE"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))
    triggered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_alerts_user_id", "user_id"),
        Index("ix_alerts_status", "status"),
    )
