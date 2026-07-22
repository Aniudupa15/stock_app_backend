"""add password_hash + not-null email to users, create refresh_tokens table

Revision ID: 0011
Revises: 0010
Create Date: 2026-07-22
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None

DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000001"
DEFAULT_USER_PLACEHOLDER_EMAIL = "default-user@local.invalid"


def upgrade() -> None:
    op.add_column("users", sa.Column("password_hash", sa.String(60), nullable=True))

    # The seeded default-user row (migration 0006) has email=NULL - back it
    # to a real placeholder before making the column NOT NULL. Its
    # password_hash stays NULL: AuthService.login treats that as "this
    # account has no password set", not an error - it simply can't
    # authenticate, unlike real registered users.
    op.execute(
        f"UPDATE users SET email = '{DEFAULT_USER_PLACEHOLDER_EMAIL}' "
        f"WHERE id = '{DEFAULT_USER_ID}' AND email IS NULL"
    )
    op.alter_column("users", "email", existing_type=sa.String(320), nullable=False)

    op.create_table(
        "refresh_tokens",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("token_hash", name="uq_refresh_tokens_token_hash"),
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_refresh_tokens_user_id", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")
    op.alter_column("users", "email", existing_type=sa.String(320), nullable=True)
    op.drop_column("users", "password_hash")
