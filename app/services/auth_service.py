import uuid
from datetime import UTC, datetime, timedelta

from app.core.config import Settings
from app.core.exceptions import EmailAlreadyRegisteredError, InvalidCredentialsError, InvalidRefreshTokenError
from app.core.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from app.domain.ports import RefreshTokenRepositoryPort, UserRepositoryPort
from app.schemas.auth import TokenPairOut


class AuthService:
    def __init__(
        self,
        user_repository: UserRepositoryPort,
        refresh_token_repository: RefreshTokenRepositoryPort,
        settings: Settings,
    ):
        self._user_repository = user_repository
        self._refresh_token_repository = refresh_token_repository
        self._settings = settings

    async def register(self, email: str, password: str, display_name: str) -> TokenPairOut:
        existing = await self._user_repository.get_by_email(email)
        if existing is not None:
            raise EmailAlreadyRegisteredError(email)

        user = await self._user_repository.create(email, hash_password(password), display_name)
        return await self._issue_tokens(user.id)

    async def login(self, email: str, password: str) -> TokenPairOut:
        user = await self._user_repository.get_by_email(email)
        if user is None or user.password_hash is None or not verify_password(password, user.password_hash):
            raise InvalidCredentialsError()
        return await self._issue_tokens(user.id)

    async def refresh(self, raw_refresh_token: str) -> TokenPairOut:
        stored = await self._refresh_token_repository.get_by_hash(hash_refresh_token(raw_refresh_token))
        if stored is None or stored.revoked_at is not None or stored.expires_at < datetime.now(UTC):
            raise InvalidRefreshTokenError()

        # Rotation: the presented token is single-use - revoke it before
        # issuing a replacement, so a stolen-and-replayed refresh token
        # can't be used again even if the legitimate client already rotated.
        await self._refresh_token_repository.revoke(stored.id)
        return await self._issue_tokens(stored.user_id)

    async def logout(self, raw_refresh_token: str) -> None:
        stored = await self._refresh_token_repository.get_by_hash(hash_refresh_token(raw_refresh_token))
        if stored is not None and stored.revoked_at is None:
            await self._refresh_token_repository.revoke(stored.id)

    async def _issue_tokens(self, user_id: uuid.UUID) -> TokenPairOut:
        access_token = create_access_token(user_id, self._settings)
        raw_refresh_token, refresh_token_hash = generate_refresh_token()
        expires_at = datetime.now(UTC) + timedelta(days=self._settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
        await self._refresh_token_repository.create(user_id, refresh_token_hash, expires_at)
        return TokenPairOut(access_token=access_token, refresh_token=raw_refresh_token)
