"""enable pg_trgm and add fuzzy search indexes

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-21
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE INDEX ix_stocks_name_trgm ON stocks USING GIN (name gin_trgm_ops)")
    op.execute("CREATE INDEX ix_stocks_symbol_trgm ON stocks USING GIN (symbol gin_trgm_ops)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_stocks_symbol_trgm")
    op.execute("DROP INDEX IF EXISTS ix_stocks_name_trgm")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
