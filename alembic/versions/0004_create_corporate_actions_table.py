"""create corporate_actions table

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-21
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "corporate_actions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("stock_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("purpose", sa.String(length=512), nullable=False),
        sa.Column("face_value", sa.Numeric(10, 2), nullable=True),
        sa.Column("ex_date", sa.Date(), nullable=True),
        sa.Column("record_date", sa.Date(), nullable=True),
        sa.Column("book_closure_start", sa.Date(), nullable=True),
        sa.Column("book_closure_end", sa.Date(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["stock_id"], ["stocks.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("stock_id", "purpose", "ex_date", name="uq_corporate_actions_stock_purpose_exdate"),
    )
    op.create_index("ix_corporate_actions_stock_id", "corporate_actions", ["stock_id"])


def downgrade() -> None:
    op.drop_index("ix_corporate_actions_stock_id", table_name="corporate_actions")
    op.drop_table("corporate_actions")
