"""create news_articles table

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-22
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None

_news_category = postgresql.ENUM("MARKET", "COMPANY", "ECONOMY", "REGULATION", "SECTOR", name="news_category")
_news_category_column = postgresql.ENUM(
    "MARKET", "COMPANY", "ECONOMY", "REGULATION", "SECTOR", name="news_category", create_type=False
)


def upgrade() -> None:
    _news_category.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "news_articles",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("headline", sa.String(512), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("source", sa.String(128), nullable=False),
        sa.Column("url", sa.String(1024), nullable=False),
        sa.Column("category", _news_category_column, nullable=False),
        sa.Column("related_symbols", postgresql.ARRAY(sa.String(32)), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("url", name="uq_news_articles_url"),
    )
    op.create_index("ix_news_articles_published_at", "news_articles", ["published_at"])


def downgrade() -> None:
    op.drop_index("ix_news_articles_published_at", table_name="news_articles")
    op.drop_table("news_articles")
    _news_category.drop(op.get_bind(), checkfirst=True)
