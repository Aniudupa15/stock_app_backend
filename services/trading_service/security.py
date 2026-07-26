"""At-rest encryption for broker API credentials (Fernet / AES-128-CBC+HMAC).

Broker `api_secret` and `access_token` must never be stored in plaintext
(Phase 1 §6). This wraps Fernet with a key from `TRADING_ENCRYPTION_KEY`,
falling back to a key derived from `JWT_SECRET_KEY` for dev/tests only.
"""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet

from app.core.config import Settings


def _fernet_key(settings: Settings) -> bytes:
    configured = settings.TRADING_ENCRYPTION_KEY.strip()
    if configured:
        # Assume a proper Fernet key (44-char urlsafe-base64 of 32 bytes).
        return configured.encode()
    # Dev/test fallback: derive a valid 32-byte Fernet key from the JWT secret.
    digest = hashlib.sha256(settings.JWT_SECRET_KEY.encode()).digest()
    return base64.urlsafe_b64encode(digest)


class CredentialCipher:
    """Encrypt/decrypt short secrets to/from bytes for BYTEA columns."""

    def __init__(self, settings: Settings) -> None:
        self._fernet = Fernet(_fernet_key(settings))

    def encrypt(self, plaintext: str) -> bytes:
        return self._fernet.encrypt(plaintext.encode())

    def decrypt(self, token: bytes) -> str:
        return self._fernet.decrypt(bytes(token)).decode()
