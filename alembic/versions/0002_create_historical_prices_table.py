"""create historical_prices table

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-17
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "historical_prices",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("stock_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("open", sa.Numeric(12, 2), nullable=False),
        sa.Column("high", sa.Numeric(12, 2), nullable=False),
        sa.Column("low", sa.Numeric(12, 2), nullable=False),
        sa.Column("close", sa.Numeric(12, 2), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["stock_id"], ["stocks.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("stock_id", "trade_date", name="uq_historical_prices_stock_date"),
    )
    op.create_index(
        "ix_hist_stock_date", "historical_prices", ["stock_id", sa.text("trade_date DESC")]
    )


def downgrade() -> None:
    op.drop_index("ix_hist_stock_date", table_name="historical_prices")
    op.drop_table("historical_prices")
