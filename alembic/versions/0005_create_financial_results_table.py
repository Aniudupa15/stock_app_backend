"""create financial_results table

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-21
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "financial_results",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("stock_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("period_start", sa.Date(), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("consolidated", sa.Boolean(), nullable=False),
        sa.Column("revenue", sa.Numeric(20, 2), nullable=True),
        sa.Column("profit", sa.Numeric(20, 2), nullable=True),
        sa.Column("eps_basic", sa.Numeric(10, 4), nullable=True),
        sa.Column("eps_diluted", sa.Numeric(10, 4), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["stock_id"], ["stocks.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "stock_id", "period_end", "consolidated", name="uq_financial_results_stock_period_consolidated"
        ),
    )
    op.create_index("ix_financial_results_stock_id", "financial_results", ["stock_id"])


def downgrade() -> None:
    op.drop_index("ix_financial_results_stock_id", table_name="financial_results")
    op.drop_table("financial_results")
