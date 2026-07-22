from app.domain.entities import StockMasterRecord
from app.repositories.stock_repository import SqlAlchemyStockRepository


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


async def test_create_list_and_delete_alert(app_client, db_session):
    client, _ = app_client
    await _seed_stock(db_session, "RELIANCE")

    create_resp = await client.post(
        "/api/v1/alerts", json={"symbol": "RELIANCE", "alert_type": "PRICE_ABOVE", "condition": {"price": "1500"}}
    )
    assert create_resp.status_code == 201
    body = create_resp.json()
    assert body["symbol"] == "RELIANCE"
    assert body["status"] == "ACTIVE"
    alert_id = body["id"]

    list_resp = await client.get("/api/v1/alerts")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1

    delete_resp = await client.delete(f"/api/v1/alerts/{alert_id}")
    assert delete_resp.status_code == 204

    list_after = await client.get("/api/v1/alerts")
    assert list_after.json() == []


async def test_create_alert_unknown_symbol_returns_404(app_client):
    client, _ = app_client
    resp = await client.post(
        "/api/v1/alerts", json={"symbol": "DOESNOTEXIST", "alert_type": "PRICE_ABOVE", "condition": {"price": "100"}}
    )
    assert resp.status_code == 404


async def test_create_alert_missing_condition_key_returns_422(app_client, db_session):
    client, _ = app_client
    await _seed_stock(db_session, "RELIANCE")

    resp = await client.post(
        "/api/v1/alerts", json={"symbol": "RELIANCE", "alert_type": "PRICE_ABOVE", "condition": {}}
    )

    assert resp.status_code == 422
