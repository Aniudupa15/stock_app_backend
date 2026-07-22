from datetime import date
from decimal import Decimal

from app.domain.entities import StockMasterRecord
from app.repositories.stock_repository import SqlAlchemyStockRepository


async def test_search_returns_db_backed_results(app_client, db_session):
    client, _ = app_client
    repo = SqlAlchemyStockRepository(db_session)
    await repo.upsert_universe(
        [
            StockMasterRecord(
                symbol="WIPRO",
                isin=None,
                name="Wipro Limited",
                series="EQ",
                listing_date=date(1946, 1, 1),
                face_value=Decimal("2.00"),
            )
        ]
    )
    await db_session.commit()

    resp = await client.get("/api/v1/stocks/search", params={"q": "wipro"})

    assert resp.status_code == 200
    body = resp.json()
    assert any(r["symbol"] == "WIPRO" for r in body)


async def test_search_empty_query_rejected(app_client):
    client, _ = app_client
    resp = await client.get("/api/v1/stocks/search", params={"q": ""})
    assert resp.status_code == 422
