"""create the trading schema and all 16 trading-platform tables

Revision ID: 0016
Revises: 0015
Create Date: 2026-07-26

Tables are created from the verified SQLAlchemy metadata (all objects under the
`trading.` schema), so the applied DDL matches the models exactly. This is a
one-shot "create the whole trading schema" migration; subsequent changes to
individual trading tables should be normal explicit migrations.
"""

# Importing the models registers every trading table on Base.metadata.
import services.trading_service.persistence.models  # noqa: F401,E402
from alembic import op
from app.models.base import Base

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None

_SCHEMA = "trading"


def _trading_tables():
    # Dependency-ordered, filtered to the trading schema only.
    return [t for t in Base.metadata.sorted_tables if t.schema == _SCHEMA]


def upgrade() -> None:
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {_SCHEMA}")
    bind = op.get_bind()
    for table in _trading_tables():
        table.create(bind, checkfirst=True)


def downgrade() -> None:
    op.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
