"""create intraday_signal_snapshots and long_term_signal_snapshots tables

Revision ID: 0015
Revises: 0014
Create Date: 2026-07-24
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "intraday_signal_snapshots",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("stock_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("signal", sa.String(10), nullable=False),
        sa.Column("confidence", sa.Numeric(5, 2), nullable=False),
        sa.Column("entry_price", sa.Numeric(12, 2), nullable=True),
        sa.Column("target_price", sa.Numeric(12, 2), nullable=True),
        sa.Column("stop_loss", sa.Numeric(12, 2), nullable=True),
        sa.Column("reasoning", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["stock_id"], ["stocks.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("stock_id", name="uq_intraday_signal_snapshots_stock_id"),
    )
    op.create_table(
        "long_term_signal_snapshots",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("stock_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("signal", sa.String(10), nullable=False),
        sa.Column("confidence", sa.Integer(), nullable=False),
        sa.Column("risk_level", sa.String(20), nullable=False),
        sa.Column("growth_potential", sa.String(20), nullable=False),
        sa.Column("investment_tenure", sa.String(20), nullable=False),
        sa.Column("reasoning", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["stock_id"], ["stocks.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("stock_id", name="uq_long_term_signal_snapshots_stock_id"),
    )


def downgrade() -> None:
    op.drop_table("long_term_signal_snapshots")
    op.drop_table("intraday_signal_snapshots")
