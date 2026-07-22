import uuid
from datetime import UTC, datetime, timedelta

from app.repositories.refresh_token_repository import SqlAlchemyRefreshTokenRepository
from app.repositories.user_repository import SqlAlchemyUserRepository
from tests.conftest import requires_docker

pytestmark = requires_docker


async def _seed_user(db_session, email: str) -> uuid.UUID:
    user = await SqlAlchemyUserRepository(db_session).create(email, "hashed", "Test User")
    return user.id


async def test_create_and_get_by_hash(db_session):
    user_id = await _seed_user(db_session, "refresh-create@example.com")
    repo = SqlAlchemyRefreshTokenRepository(db_session)
    expires_at = datetime.now(UTC) + timedelta(days=30)

    created = await repo.create(user_id, "a-token-hash", expires_at)

    fetched = await repo.get_by_hash("a-token-hash")
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.revoked_at is None


async def test_get_by_hash_unknown_returns_none(db_session):
    repo = SqlAlchemyRefreshTokenRepository(db_session)

    assert await repo.get_by_hash("does-not-exist") is None


async def test_revoke_sets_revoked_at(db_session):
    user_id = await _seed_user(db_session, "refresh-revoke@example.com")
    repo = SqlAlchemyRefreshTokenRepository(db_session)
    created = await repo.create(user_id, "revoke-me-hash", datetime.now(UTC) + timedelta(days=30))

    await repo.revoke(created.id)

    fetched = await repo.get_by_hash("revoke-me-hash")
    assert fetched.revoked_at is not None
