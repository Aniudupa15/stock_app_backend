from app.domain.entities import StockMasterRecord
from app.repositories.stock_repository import SqlAlchemyStockRepository


async def test_corporate_actions_unknown_symbol_returns_404(app_client):
    client, _ = app_client
    resp = await client.get("/api/v1/stocks/DOESNOTEXIST/corporate-actions")
    assert resp.status_code == 404


async def test_corporate_actions_returns_empty_list_when_none_synced_yet(app_client, db_session):
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [StockMasterRecord(symbol="INFY", isin=None, name="Infosys", series="EQ", listing_date=None, face_value=None)]
    )
    await db_session.commit()

    client, _ = app_client
    resp = await client.get("/api/v1/stocks/INFY/corporate-actions")

    assert resp.status_code == 200
    assert resp.json() == []
