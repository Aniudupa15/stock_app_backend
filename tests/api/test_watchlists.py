import uuid
from datetime import date
from decimal import Decimal

from app.domain.entities import BhavcopyRecord, StockMasterRecord
from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository


async def _seed_stock_with_price(db_session, symbol: str) -> None:
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [
            StockMasterRecord(
                symbol=symbol, isin=None, name=f"{symbol} Ltd", series="EQ", listing_date=None, face_value=None
            )
        ]
    )
    await db_session.commit()

    price_repo = SqlAlchemyHistoricalPriceRepository(db_session)
    await price_repo.bulk_upsert_bars(
        [
            BhavcopyRecord(
                symbol=symbol,
                trade_date=date(2026, 7, 21),
                open=Decimal("100"),
                high=Decimal("100"),
                low=Decimal("100"),
                close=Decimal("100"),
                volume=1000,
            )
        ]
    )


async def test_create_list_and_get_watchlist(app_client, db_session):
    client, _ = app_client
    await _seed_stock_with_price(db_session, "RELIANCE")

    create_resp = await client.post("/api/v1/watchlists", json={"name": "My Picks"})
    assert create_resp.status_code == 201
    watchlist_id = create_resp.json()["id"]

    list_resp = await client.get("/api/v1/watchlists")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1

    add_resp = await client.post(f"/api/v1/watchlists/{watchlist_id}/items", json={"symbol": "RELIANCE"})
    assert add_resp.status_code == 200
    body = add_resp.json()
    assert body["items"][0]["symbol"] == "RELIANCE"
    assert body["items"][0]["last_price"] == "100.00"

    detail_resp = await client.get(f"/api/v1/watchlists/{watchlist_id}")
    assert detail_resp.status_code == 200
    assert len(detail_resp.json()["items"]) == 1


async def test_add_unknown_symbol_returns_404(app_client):
    client, _ = app_client
    create_resp = await client.post("/api/v1/watchlists", json={"name": "My Picks"})
    watchlist_id = create_resp.json()["id"]

    resp = await client.post(f"/api/v1/watchlists/{watchlist_id}/items", json={"symbol": "DOESNOTEXIST"})

    assert resp.status_code == 404


async def test_unknown_watchlist_returns_404(app_client):
    client, _ = app_client
    resp = await client.get(f"/api/v1/watchlists/{uuid.uuid4()}")
    assert resp.status_code == 404


async def test_remove_item_and_delete_watchlist(app_client, db_session):
    client, _ = app_client
    await _seed_stock_with_price(db_session, "TCS")

    create_resp = await client.post("/api/v1/watchlists", json={"name": "My Picks"})
    watchlist_id = create_resp.json()["id"]
    await client.post(f"/api/v1/watchlists/{watchlist_id}/items", json={"symbol": "TCS"})

    remove_resp = await client.delete(f"/api/v1/watchlists/{watchlist_id}/items/TCS")
    assert remove_resp.status_code == 200
    assert remove_resp.json()["items"] == []

    delete_resp = await client.delete(f"/api/v1/watchlists/{watchlist_id}")
    assert delete_resp.status_code == 204

    get_resp = await client.get(f"/api/v1/watchlists/{watchlist_id}")
    assert get_resp.status_code == 404
