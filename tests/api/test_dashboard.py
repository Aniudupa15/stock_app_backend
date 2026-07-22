from datetime import date
from decimal import Decimal

from app.domain.entities import BhavcopyRecord, StockMasterRecord
from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository


async def test_dashboard_returns_composed_sections(app_client, db_session):
    client, _ = app_client
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [StockMasterRecord(symbol="UP", isin=None, name="Up Ltd", series="EQ", listing_date=None, face_value=None)]
    )
    await db_session.commit()

    price_repo = SqlAlchemyHistoricalPriceRepository(db_session)
    await price_repo.bulk_upsert_bars(
        [
            BhavcopyRecord(
                symbol="UP",
                trade_date=date(2026, 7, 20),
                open=Decimal("100"),
                high=Decimal("100"),
                low=Decimal("100"),
                close=Decimal("100"),
                volume=500,
            ),
            BhavcopyRecord(
                symbol="UP",
                trade_date=date(2026, 7, 21),
                open=Decimal("110"),
                high=Decimal("110"),
                low=Decimal("110"),
                close=Decimal("110"),
                volume=500,
            ),
        ]
    )

    resp = await client.get("/api/v1/dashboard")

    assert resp.status_code == 200
    body = resp.json()
    assert "market_status" in body
    assert "indices" in body
    assert body["gainers"][0]["symbol"] == "UP"
    assert body["most_active"][0]["symbol"] == "UP"
    assert isinstance(body["notes"], list)
    assert len(body["notes"]) == 3
