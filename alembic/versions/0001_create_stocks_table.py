"""create stocks table

Revision ID: 0001
Revises:
Create Date: 2026-07-17
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

instrument_type_enum = sa.Enum("EQUITY", "ETF", "REIT", "INVIT", name="instrument_type")


def upgrade() -> None:
    op.create_table(
        "stocks",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("isin", sa.String(length=12), nullable=True),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("series", sa.String(length=8), nullable=True),
        sa.Column("sector", sa.String(length=128), nullable=True),
        sa.Column("industry", sa.String(length=128), nullable=True),
        sa.Column("instrument_type", instrument_type_enum, nullable=False, server_default="EQUITY"),
        sa.Column("listing_date", sa.Date(), nullable=True),
        sa.Column("face_value", sa.Numeric(10, 2), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", name="uq_stocks_symbol"),
        sa.UniqueConstraint("isin", name="uq_stocks_isin"),
    )
    op.create_index("ix_stocks_symbol", "stocks", ["symbol"])
    op.create_index("ix_stocks_isin", "stocks", ["isin"])
    op.create_index("ix_stocks_name_lower", "stocks", [sa.text("lower(name)")])
    op.create_index("ix_stocks_is_active", "stocks", ["is_active"])


def downgrade() -> None:
    op.drop_index("ix_stocks_is_active", table_name="stocks")
    op.drop_index("ix_stocks_name_lower", table_name="stocks")
    op.drop_index("ix_stocks_isin", table_name="stocks")
    op.drop_index("ix_stocks_symbol", table_name="stocks")
    op.drop_table("stocks")
    instrument_type_enum.drop(op.get_bind(), checkfirst=True)
