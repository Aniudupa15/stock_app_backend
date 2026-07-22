import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.api import deps
from app.core.auth import DEFAULT_USER_ID
from app.core.config import get_settings
from app.core.security import create_access_token
from app.main import create_app
from tests.conftest import FakeCache, FakeStockDataProvider

# A fixture family that yields THREE variants of an unauthenticated-to-various-degrees
# client, to prove `get_current_user_id` actually rejects bad input rather than
# `app_client`'s valid token just happening to make every test pass.


@pytest_asyncio.fixture
async def no_auth_client(db_session, sample_quote):
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


async def test_protected_endpoint_without_token_returns_403(no_auth_client):
    # FastAPI's HTTPBearer returns 403 (not 401) when the Authorization
    # header is missing entirely - this is standard FastAPI/Starlette
    # behavior for a missing security scheme, distinct from a present-but-
    # invalid token (which our own code maps to 401).
    resp = await no_auth_client.post("/api/v1/watchlists", json={"name": "My Picks"})
    assert resp.status_code == 403


async def test_protected_endpoint_with_garbage_token_returns_401(no_auth_client):
    no_auth_client.headers["Authorization"] = "Bearer not-a-real-jwt"
    resp = await no_auth_client.post("/api/v1/watchlists", json={"name": "My Picks"})
    assert resp.status_code == 401


async def test_protected_endpoint_with_valid_token_succeeds(no_auth_client):
    # DEFAULT_USER_ID, not a fresh random UUID - watchlists.user_id has a
    # real FK to users.id, and only the seeded default row is guaranteed to
    # exist without going through /auth/register first.
    token = create_access_token(DEFAULT_USER_ID, get_settings())
    no_auth_client.headers["Authorization"] = f"Bearer {token}"

    resp = await no_auth_client.post("/api/v1/watchlists", json={"name": "My Picks"})

    assert resp.status_code == 201
