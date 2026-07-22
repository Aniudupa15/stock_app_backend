from datetime import date, timedelta
from decimal import Decimal

from app.domain.entities import StockMasterRecord
from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository


async def test_history_returns_bars_within_range(app_client, db_session):
    client, _ = app_client
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [
            StockMasterRecord(
                symbol="RELIANCE", isin=None, name="Reliance", series="EQ", listing_date=None, face_value=None
            )
        ]
    )
    await db_session.commit()

    price_repo = SqlAlchemyHistoricalPriceRepository(db_session)
    from app.domain.entities import BhavcopyRecord

    await price_repo.bulk_upsert_bars(
        [
            BhavcopyRecord(
                symbol="RELIANCE",
                trade_date=date.today() - timedelta(days=5),
                open=Decimal("2490"),
                high=Decimal("2510"),
                low=Decimal("2480"),
                close=Decimal("2500"),
                volume=1_000_000,
            )
        ]
    )

    resp = await client.get("/api/v1/stocks/RELIANCE/history", params={"range": "1M"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "RELIANCE"
    assert body["range"] == "1M"
    assert len(body["bars"]) == 1
    assert body["bars"][0]["close"] == "2500.00"


async def test_history_unknown_symbol_returns_404(app_client):
    client, _ = app_client
    resp = await client.get("/api/v1/stocks/DOESNOTEXIST/history")
    assert resp.status_code == 404
