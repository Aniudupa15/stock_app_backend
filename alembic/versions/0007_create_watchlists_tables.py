"""create watchlists and watchlist_items tables

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-22
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "watchlists",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_watchlists_user_id", "watchlists", ["user_id"])

    op.create_table(
        "watchlist_items",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("watchlist_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stock_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("added_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["watchlist_id"], ["watchlists.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["stock_id"], ["stocks.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("watchlist_id", "stock_id", name="uq_watchlist_items_watchlist_stock"),
    )
    op.create_index("ix_watchlist_items_watchlist_id", "watchlist_items", ["watchlist_id"])


def downgrade() -> None:
    op.drop_index("ix_watchlist_items_watchlist_id", table_name="watchlist_items")
    op.drop_table("watchlist_items")
    op.drop_index("ix_watchlists_user_id", table_name="watchlists")
    op.drop_table("watchlists")
