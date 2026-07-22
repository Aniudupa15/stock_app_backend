import time
import uuid

import jwt
import pytest
from fastapi import HTTPException

from app.core.config import Settings
from app.core.security import (
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)


@pytest.fixture
def settings() -> Settings:
    return Settings(DATABASE_URL="postgresql+asyncpg://test:test@localhost/test", JWT_SECRET_KEY="test-secret")


def test_hash_password_and_verify_roundtrip():
    hashed = hash_password("correct-horse-battery-staple")

    assert hashed != "correct-horse-battery-staple"
    assert verify_password("correct-horse-battery-staple", hashed) is True


def test_verify_password_rejects_wrong_password():
    hashed = hash_password("correct-horse-battery-staple")

    assert verify_password("wrong-password", hashed) is False


def test_verify_password_rejects_malformed_hash():
    assert verify_password("anything", "not-a-real-bcrypt-hash") is False


def test_create_and_decode_access_token_roundtrip(settings):
    user_id = uuid.uuid4()

    token = create_access_token(user_id, settings)
    decoded = decode_access_token(token, settings)

    assert decoded == user_id


def test_decode_access_token_rejects_garbage(settings):
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token("not-a-jwt", settings)
    assert exc_info.value.status_code == 401


def test_decode_access_token_rejects_wrong_secret(settings):
    user_id = uuid.uuid4()
    token = create_access_token(user_id, settings)
    wrong_settings = Settings(DATABASE_URL=settings.DATABASE_URL, JWT_SECRET_KEY="a-different-secret")

    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(token, wrong_settings)
    assert exc_info.value.status_code == 401


def test_decode_access_token_rejects_expired_token(settings):
    now = int(time.time())
    payload = {"sub": str(uuid.uuid4()), "type": "access", "iat": now - 3600, "exp": now - 1}
    expired_token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(expired_token, settings)
    assert exc_info.value.status_code == 401


def test_decode_access_token_rejects_non_access_token_type(settings):
    now = int(time.time())
    payload = {"sub": str(uuid.uuid4()), "type": "refresh", "iat": now, "exp": now + 3600}
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(token, settings)
    assert exc_info.value.status_code == 401


def test_generate_refresh_token_returns_matching_raw_and_hash():
    raw, token_hash = generate_refresh_token()

    assert raw != token_hash
    assert hash_refresh_token(raw) == token_hash


def test_generate_refresh_token_is_random():
    raw1, _ = generate_refresh_token()
    raw2, _ = generate_refresh_token()

    assert raw1 != raw2
