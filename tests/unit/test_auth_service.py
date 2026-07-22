import uuid
from datetime import UTC, datetime

import pytest

from app.core.exceptions import EmailAlreadyRegisteredError, InvalidCredentialsError, InvalidRefreshTokenError
from app.domain.entities import User
from app.services.auth_service import AuthService
from tests.conftest import FakeRefreshTokenRepository, FakeUserRepository


@pytest.fixture
def service(settings) -> tuple[AuthService, FakeUserRepository, FakeRefreshTokenRepository]:
    user_repo = FakeUserRepository()
    refresh_repo = FakeRefreshTokenRepository()
    return AuthService(user_repo, refresh_repo, settings), user_repo, refresh_repo


async def test_register_creates_user_and_issues_tokens(service):
    svc, user_repo, _ = service

    tokens = await svc.register("new@example.com", "password123", "New User")

    assert tokens.access_token
    assert tokens.refresh_token
    stored = await user_repo.get_by_email("new@example.com")
    assert stored is not None
    assert stored.password_hash != "password123"


async def test_register_duplicate_email_raises(service):
    svc, *_ = service
    await svc.register("taken@example.com", "password123", "First User")

    with pytest.raises(EmailAlreadyRegisteredError):
        await svc.register("taken@example.com", "password123", "Second User")


async def test_login_succeeds_with_correct_password(service):
    svc, *_ = service
    await svc.register("user@example.com", "password123", "User")

    tokens = await svc.login("user@example.com", "password123")

    assert tokens.access_token
    assert tokens.refresh_token


async def test_login_fails_with_wrong_password(service):
    svc, *_ = service
    await svc.register("user@example.com", "password123", "User")

    with pytest.raises(InvalidCredentialsError):
        await svc.login("user@example.com", "wrong-password")


async def test_login_fails_for_unknown_email(service):
    svc, *_ = service

    with pytest.raises(InvalidCredentialsError):
        await svc.login("nobody@example.com", "password123")


async def test_login_fails_for_user_with_no_password_hash(service):
    svc, user_repo, _ = service
    passwordless_user = User(
        id=uuid.uuid4(), email="nopass@example.com", display_name="X", password_hash=None, created_at=datetime.now(UTC)
    )
    user_repo.users_by_id[passwordless_user.id] = passwordless_user

    with pytest.raises(InvalidCredentialsError):
        await svc.login("nopass@example.com", "anything")


async def test_refresh_rotates_token_and_invalidates_old_one(service):
    # Note: access tokens aren't asserted distinct here - two JWTs issued
    # for the same user within the same second are byte-identical by
    # design (iat/exp truncate to whole seconds), that's not a bug. Only
    # the refresh token is required to be single-use/rotated.
    svc, *_ = service
    initial = await svc.register("user@example.com", "password123", "User")

    rotated = await svc.refresh(initial.refresh_token)

    assert rotated.refresh_token != initial.refresh_token
    assert rotated.access_token

    with pytest.raises(InvalidRefreshTokenError):
        await svc.refresh(initial.refresh_token)


async def test_refresh_rejects_unknown_token(service):
    svc, *_ = service

    with pytest.raises(InvalidRefreshTokenError):
        await svc.refresh("not-a-real-token")


async def test_logout_revokes_refresh_token(service):
    svc, *_ = service
    tokens = await svc.register("user@example.com", "password123", "User")

    await svc.logout(tokens.refresh_token)

    with pytest.raises(InvalidRefreshTokenError):
        await svc.refresh(tokens.refresh_token)


async def test_logout_unknown_token_is_a_no_op(service):
    svc, *_ = service

    await svc.logout("not-a-real-token")
