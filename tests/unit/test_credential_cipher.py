"""Unit tests for broker-credential encryption. Pure - no DB, no network."""

from app.core.config import Settings
from services.trading_service.security import CredentialCipher


def _cipher(key: str = "") -> CredentialCipher:
    return CredentialCipher(Settings(TRADING_ENCRYPTION_KEY=key, JWT_SECRET_KEY="unit-test-secret"))


def test_encrypt_decrypt_round_trip():
    cipher = _cipher()
    token = cipher.encrypt("my-api-secret")
    assert isinstance(token, bytes)
    assert token != b"my-api-secret"  # actually encrypted
    assert cipher.decrypt(token) == "my-api-secret"


def test_ciphertext_is_nondeterministic():
    cipher = _cipher()
    assert cipher.encrypt("same") != cipher.encrypt("same")  # Fernet nonce/timestamp


def test_derived_key_is_stable_for_same_secret():
    # Two ciphers derived from the same JWT secret can decrypt each other.
    a = _cipher()
    b = _cipher()
    assert b.decrypt(a.encrypt("shared")) == "shared"


def test_explicit_fernet_key_is_used():
    from cryptography.fernet import Fernet

    key = Fernet.generate_key().decode()
    cipher = _cipher(key)
    assert cipher.decrypt(cipher.encrypt("x")) == "x"
