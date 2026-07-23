import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.api import deps
from app.main import create_app
from tests.conftest import FakeCache, FakeStockDataProvider

# This file deliberately does NOT use the shared `app_client` fixture, since
# that fixture pre-authenticates every request - auth itself is exactly what
# these tests exercise. `auth_client` mirrors it minus the bearer token.


@pytest_asyncio.fixture
async def auth_client(db_session, sample_quote):
    app = create_app()

    async def _get_db_session():
        yield db_session

    app.dependency_overrides[deps.get_db_session] = _get_db_session
    app.dependency_overrides[deps.get_nse_provider] = lambda: FakeStockDataProvider(quotes={"RELIANCE": sample_quote})
    app.dependency_overrides[deps.get_cache] = lambda: FakeCache()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


async def test_register_returns_tokens(auth_client):
    resp = await auth_client.post(
        "/api/v1/auth/register",
        json={"email": "alice@example.com", "password": "password123", "display_name": "Alice"},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["token_type"] == "bearer"


async def test_register_duplicate_email_returns_409(auth_client):
    await auth_client.post(
        "/api/v1/auth/register",
        json={"email": "bob@example.com", "password": "password123", "display_name": "Bob"},
    )

    resp = await auth_client.post(
        "/api/v1/auth/register",
        json={"email": "bob@example.com", "password": "password123", "display_name": "Bob Again"},
    )

    assert resp.status_code == 409


async def test_login_with_correct_credentials_returns_tokens(auth_client):
    await auth_client.post(
        "/api/v1/auth/register",
        json={"email": "carol@example.com", "password": "password123", "display_name": "Carol"},
    )

    resp = await auth_client.post("/api/v1/auth/login", json={"email": "carol@example.com", "password": "password123"})

    assert resp.status_code == 200
    assert resp.json()["access_token"]


async def test_login_with_wrong_password_returns_401(auth_client):
    await auth_client.post(
        "/api/v1/auth/register",
        json={"email": "dave@example.com", "password": "password123", "display_name": "Dave"},
    )

    resp = await auth_client.post("/api/v1/auth/login", json={"email": "dave@example.com", "password": "wrong"})

    assert resp.status_code == 401


async def test_login_unknown_email_returns_401(auth_client):
    resp = await auth_client.post("/api/v1/auth/login", json={"email": "nobody@example.com", "password": "password123"})

    assert resp.status_code == 401


async def test_refresh_rotates_and_invalidates_old_token(auth_client):
    register_resp = await auth_client.post(
        "/api/v1/auth/register",
        json={"email": "erin@example.com", "password": "password123", "display_name": "Erin"},
    )
    initial_refresh = register_resp.json()["refresh_token"]

    refresh_resp = await auth_client.post("/api/v1/auth/refresh", json={"refresh_token": initial_refresh})
    assert refresh_resp.status_code == 200
    assert refresh_resp.json()["refresh_token"] != initial_refresh

    reuse_resp = await auth_client.post("/api/v1/auth/refresh", json={"refresh_token": initial_refresh})
    assert reuse_resp.status_code == 401


async def test_refresh_with_garbage_token_returns_401(auth_client):
    resp = await auth_client.post("/api/v1/auth/refresh", json={"refresh_token": "not-a-real-token"})

    assert resp.status_code == 401


async def test_logout_then_refresh_returns_401(auth_client):
    register_resp = await auth_client.post(
        "/api/v1/auth/register",
        json={"email": "frank@example.com", "password": "password123", "display_name": "Frank"},
    )
    refresh_token = register_resp.json()["refresh_token"]

    logout_resp = await auth_client.post("/api/v1/auth/logout", json={"refresh_token": refresh_token})
    assert logout_resp.status_code == 204

    refresh_resp = await auth_client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert refresh_resp.status_code == 401


async def test_register_rejects_short_password(auth_client):
    resp = await auth_client.post(
        "/api/v1/auth/register",
        json={"email": "grace@example.com", "password": "short", "display_name": "Grace"},
    )

    assert resp.status_code == 422


async def test_register_rejects_invalid_email(auth_client):
    resp = await auth_client.post(
        "/api/v1/auth/register",
        json={"email": "not-an-email", "password": "password123", "display_name": "Grace"},
    )

    assert resp.status_code == 422


async def test_get_me_returns_profile(auth_client):
    register_resp = await auth_client.post(
        "/api/v1/auth/register",
        json={"email": "profile-get@example.com", "password": "password123", "display_name": "Profile Getter"},
    )
    access_token = register_resp.json()["access_token"]

    resp = await auth_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access_token}"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "profile-get@example.com"
    assert body["display_name"] == "Profile Getter"


async def test_get_me_without_token_returns_403(auth_client):
    resp = await auth_client.get("/api/v1/auth/me")
    assert resp.status_code == 403


async def test_update_me_changes_display_name(auth_client):
    register_resp = await auth_client.post(
        "/api/v1/auth/register",
        json={"email": "profile-update@example.com", "password": "password123", "display_name": "Old Name"},
    )
    access_token = register_resp.json()["access_token"]

    resp = await auth_client.patch(
        "/api/v1/auth/me", json={"display_name": "New Name"}, headers={"Authorization": f"Bearer {access_token}"}
    )

    assert resp.status_code == 200
    assert resp.json()["display_name"] == "New Name"
    assert resp.json()["email"] == "profile-update@example.com"


async def test_update_me_rejects_email_taken_by_another_user(auth_client):
    await auth_client.post(
        "/api/v1/auth/register",
        json={"email": "taken-profile@example.com", "password": "password123", "display_name": "First"},
    )
    register_resp = await auth_client.post(
        "/api/v1/auth/register",
        json={"email": "second-profile@example.com", "password": "password123", "display_name": "Second"},
    )
    access_token = register_resp.json()["access_token"]

    resp = await auth_client.patch(
        "/api/v1/auth/me",
        json={"email": "taken-profile@example.com"},
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert resp.status_code == 409
