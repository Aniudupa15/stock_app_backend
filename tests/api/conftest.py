import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.api import deps
from app.core.auth import DEFAULT_USER_ID
from app.core.config import get_settings
from app.core.security import create_access_token
from app.main import create_app
from tests.conftest import FakeCache, FakeStockDataProvider

# No pytestmark here: a conftest.py's module-level `pytestmark` does not
# propagate to sibling test modules in the same directory. Docker-gating
# happens implicitly via `postgres_url` calling pytest.skip() when Docker
# is unavailable (see tests/conftest.py) - every test using `app_client`
# (which depends on `db_session` -> `postgres_url`) skips automatically.


@pytest_asyncio.fixture
async def app_client(db_session, sample_quote):
    """Full FastAPI app, real DB (testcontainers Postgres), NSE provider + cache
    faked at the DI boundary - exercises the real router -> service -> repository chain.

    The client carries a real, validly-signed access token for the seeded
    default user by default (auth itself is exercised separately in
    test_auth.py/test_auth_required.py) - this is what keeps every existing
    endpoint test passing unmodified after Phase 5 replaced the single-user
    bypass with real JWT verification.
    """
    app = create_app()

    fake_provider = FakeStockDataProvider(quotes={"RELIANCE": sample_quote})
    fake_cache = FakeCache()

    async def _get_db_session():
        yield db_session

    app.dependency_overrides[deps.get_db_session] = _get_db_session
    app.dependency_overrides[deps.get_nse_provider] = lambda: fake_provider
    app.dependency_overrides[deps.get_cache] = lambda: fake_cache

    access_token = create_access_token(DEFAULT_USER_ID, get_settings())
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {access_token}"}
    ) as client:
        yield client, fake_provider

    app.dependency_overrides.clear()
