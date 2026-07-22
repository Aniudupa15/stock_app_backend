"""Password hashing and JWT/refresh-token helpers. Pure functions (plus one
settings-dependent pair) - no DB, no FastAPI - so they're trivially unit
testable and reusable from both `AuthService` and `app/api/deps.py`.

Access tokens are stateless JWTs (fast, no DB hit to verify). Refresh tokens
are opaque random strings whose SHA-256 hash is the only thing ever
persisted (same principle as password hashing - never store the usable
secret itself) - this makes them revocable and rotatable, which a
self-contained JWT refresh token couldn't be without a server-side denylist
anyway.
"""

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from fastapi import HTTPException, status

from app.core.config import Settings

_ACCESS_TOKEN_TYPE = "access"

# bcrypt silently truncates passwords beyond 72 bytes - reject rather than
# let a user's password be effectively shortened without their knowledge.
MAX_PASSWORD_LENGTH = 72


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(user_id: uuid.UUID, settings: Settings) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "type": _ACCESS_TOKEN_TYPE,
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str, settings: Settings) -> uuid.UUID:
    """Raises HTTPException(401) directly - auth is a transport-layer
    concern here, not a domain one, so it doesn't need a domain exception
    translated by a handler the way StockNotFoundError etc. do.
    """
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.InvalidTokenError:
        raise unauthorized from None

    if payload.get("type") != _ACCESS_TOKEN_TYPE:
        raise unauthorized
    try:
        return uuid.UUID(payload["sub"])
    except (KeyError, ValueError):
        raise unauthorized from None


def generate_refresh_token() -> tuple[str, str]:
    """Returns (raw_token, sha256_hash). Only the hash is ever persisted."""
    raw = secrets.token_urlsafe(32)
    return raw, hash_refresh_token(raw)


def hash_refresh_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
