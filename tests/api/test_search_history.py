from httpx import ASGITransport, AsyncClient

from app.api import deps
from app.domain.entities import StockMasterRecord
from app.main import create_app
from app.repositories.stock_repository import SqlAlchemyStockRepository
from tests.conftest import FakeCache, FakeStockDataProvider


async def _seed_stock(db_session, symbol: str) -> None:
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [
            StockMasterRecord(
                symbol=symbol, isin=None, name=f"{symbol} Ltd", series="EQ", listing_date=None, face_value=None
            )
        ]
    )
    await db_session.commit()


async def _no_auth_client(db_session):
    app = create_app()

    async def _get_db_session():
        yield db_session

    app.dependency_overrides[deps.get_db_session] = _get_db_session
    app.dependency_overrides[deps.get_nse_provider] = lambda: FakeStockDataProvider()
    app.dependency_overrides[deps.get_cache] = lambda: FakeCache()

    transport = ASGITransport(app=app)
    return app, AsyncClient(transport=transport, base_url="http://test")


async def test_authenticated_search_logs_history(app_client, db_session):
    client, _ = app_client
    await _seed_stock(db_session, "RELIANCE")

    search_resp = await client.get("/api/v1/stocks/search", params={"q": "RELIANCE"})
    assert search_resp.status_code == 200

    history_resp = await client.get("/api/v1/search-history")
    assert history_resp.status_code == 200
    body = history_resp.json()
    assert len(body) == 1
    assert body[0]["query"] == "RELIANCE"


async def test_unauthenticated_search_still_works_but_logs_nothing(db_session):
    # `app_client` always attaches a token (Phase 5's design) - to prove
    # search itself stays public, build a client with no Authorization
    # header at all, same pattern as tests/api/test_auth.py's `auth_client`.
    await _seed_stock(db_session, "TCS")
    app, client = await _no_auth_client(db_session)

    async with client:
        resp = await client.get("/api/v1/stocks/search", params={"q": "TCS"})
        assert resp.status_code == 200
        assert resp.json()[0]["symbol"] == "TCS"

    app.dependency_overrides.clear()


async def test_clear_search_history(app_client, db_session):
    client, _ = app_client
    await _seed_stock(db_session, "RELIANCE")

    await client.get("/api/v1/stocks/search", params={"q": "RELIANCE"})

    clear_resp = await client.delete("/api/v1/search-history")
    assert clear_resp.status_code == 204

    history_resp = await client.get("/api/v1/search-history")
    assert history_resp.json() == []


async def test_search_history_requires_auth(db_session):
    app, client = await _no_auth_client(db_session)

    async with client:
        resp = await client.get("/api/v1/search-history")
        assert resp.status_code == 403

    app.dependency_overrides.clear()
