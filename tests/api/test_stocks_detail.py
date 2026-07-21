from datetime import date
from decimal import Decimal

from app.core.exceptions import ProviderUnavailableError
from app.domain.entities import StockMasterRecord
from app.repositories.stock_repository import SqlAlchemyStockRepository


async def test_detail_returns_static_info_and_live_quote(app_client, db_session):
    client, _ = app_client
    repo = SqlAlchemyStockRepository(db_session)
    await repo.upsert_universe(
        [
            StockMasterRecord(
                symbol="RELIANCE",
                isin="INE002A01018",
                name="Reliance Industries Limited",
                series="EQ",
                listing_date=date(1995, 1, 1),
                face_value=Decimal("10.00"),
            )
        ]
    )
    await db_session.commit()

    resp = await client.get("/api/v1/stocks/RELIANCE")

    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "RELIANCE"
    assert body["quote"] is not None
    assert body["quote_unavailable_reason"] is None


async def test_detail_degrades_gracefully_when_nse_unavailable(app_client, db_session):
    client, fake_provider = app_client
    fake_provider.fail_with = ProviderUnavailableError("NSE", "circuit open")

    repo = SqlAlchemyStockRepository(db_session)
    await repo.upsert_universe(
        [
            StockMasterRecord(
                symbol="TCS", isin=None, name="Tata Consultancy Services", series="EQ", listing_date=None, face_value=None
            )
        ]
    )
    await db_session.commit()

    resp = await client.get("/api/v1/stocks/TCS")

    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "TCS"
    assert body["quote"] is None
    assert body["quote_unavailable_reason"] is not None


async def test_detail_unknown_symbol_returns_404(app_client):
    client, _ = app_client
    resp = await client.get("/api/v1/stocks/DOESNOTEXIST")
    assert resp.status_code == 404
