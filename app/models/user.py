import uuid

from sqlalchemy import String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class UserModel(Base, TimestampMixin):
    """Real user records as of Phase 5's JWT auth. The original seeded
    default-user row (see migration 0006) keeps `password_hash=NULL` -
    `AuthService.login` treats that as "this account has no password set",
    not an error; it just can't authenticate, unlike real registered users.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    password_hash: Mapped[str | None] = mapped_column(String(60), nullable=True)
