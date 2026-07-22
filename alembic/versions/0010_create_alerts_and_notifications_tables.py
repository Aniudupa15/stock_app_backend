"""create alerts and notifications tables

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-22
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None

_ALERT_TYPES = (
    "PRICE_ABOVE",
    "PRICE_BELOW",
    "PERCENT_CHANGE_ABOVE",
    "PERCENT_CHANGE_BELOW",
    "RSI_ABOVE",
    "RSI_BELOW",
    "VOLUME_SPIKE",
    "NEW_52_WEEK_HIGH",
    "NEW_52_WEEK_LOW",
)
_alert_type = postgresql.ENUM(*_ALERT_TYPES, name="alert_type")
_alert_type_column = postgresql.ENUM(*_ALERT_TYPES, name="alert_type", create_type=False)

_alert_status = postgresql.ENUM("ACTIVE", "TRIGGERED", "CANCELLED", name="alert_status")
_alert_status_column = postgresql.ENUM("ACTIVE", "TRIGGERED", "CANCELLED", name="alert_status", create_type=False)


def upgrade() -> None:
    _alert_type.create(op.get_bind(), checkfirst=True)
    _alert_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stock_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("alert_type", _alert_type_column, nullable=False),
        sa.Column("condition", postgresql.JSONB(), nullable=False),
        sa.Column("status", _alert_status_column, nullable=False, server_default="ACTIVE"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["stock_id"], ["stocks.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_alerts_user_id", "alerts", ["user_id"])
    op.create_index("ix_alerts_status", "alerts", ["status"])

    op.create_table(
        "notifications",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("alert_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["alert_id"], ["alerts.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_notifications_user_id", "notifications", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_notifications_user_id", table_name="notifications")
    op.drop_table("notifications")
    op.drop_index("ix_alerts_status", table_name="alerts")
    op.drop_index("ix_alerts_user_id", table_name="alerts")
    op.drop_table("alerts")
    _alert_status.drop(op.get_bind(), checkfirst=True)
    _alert_type.drop(op.get_bind(), checkfirst=True)
