"""create ipo_filings table

Revision ID: 0014
Revises: 0013
Create Date: 2026-07-23
"""

import sqlalchemy as sa

from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ipo_filings",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("company_name", sa.String(256), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("price_range", sa.String(64), nullable=True),
        sa.Column("issue_size", sa.String(64), nullable=True),
        sa.Column("issue_start_date", sa.Date(), nullable=True),
        sa.Column("issue_end_date", sa.Date(), nullable=True),
        sa.Column("listing_date", sa.Date(), nullable=True),
        sa.Column("series", sa.String(8), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", name="uq_ipo_filings_symbol"),
    )


def downgrade() -> None:
    op.drop_table("ipo_filings")
