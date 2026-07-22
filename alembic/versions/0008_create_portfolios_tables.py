"""create portfolios and portfolio_transactions tables

Revision ID: 0008
Revises: 0007
Create Date: 2026-07-22
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

# `create_type=False` on the column below prevents op.create_table from
# re-emitting CREATE TYPE after the explicit create() call here - Postgres
# enum types are created once, independent of any table that references them.
_transaction_type = postgresql.ENUM("BUY", "SELL", name="transaction_type")
_transaction_type_column = postgresql.ENUM("BUY", "SELL", name="transaction_type", create_type=False)


def upgrade() -> None:
    _transaction_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "portfolios",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_portfolios_user_id", "portfolios", ["user_id"])

    op.create_table(
        "portfolio_transactions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("portfolio_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stock_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("transaction_type", _transaction_type_column, nullable=False),
        sa.Column("quantity", sa.Numeric(18, 4), nullable=False),
        sa.Column("price", sa.Numeric(12, 2), nullable=False),
        sa.Column("transaction_date", sa.Date(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["portfolio_id"], ["portfolios.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["stock_id"], ["stocks.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_portfolio_transactions_portfolio_id", "portfolio_transactions", ["portfolio_id"])


def downgrade() -> None:
    op.drop_index("ix_portfolio_transactions_portfolio_id", table_name="portfolio_transactions")
    op.drop_table("portfolio_transactions")
    op.drop_index("ix_portfolios_user_id", table_name="portfolios")
    op.drop_table("portfolios")
    _transaction_type.drop(op.get_bind(), checkfirst=True)
