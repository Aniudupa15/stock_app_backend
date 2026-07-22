from datetime import date, timedelta
from decimal import Decimal

from app.domain.entities import BhavcopyRecord, StockMasterRecord
from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository


async def test_indicators_unknown_symbol_returns_404(app_client):
    client, _ = app_client
    resp = await client.get("/api/v1/stocks/DOESNOTEXIST/indicators")
    assert resp.status_code == 404


async def test_indicators_returns_has_data_false_with_no_bars(app_client, db_session):
    client, _ = app_client
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [StockMasterRecord(symbol="WIPRO", isin=None, name="Wipro", series="EQ", listing_date=None, face_value=None)]
    )
    await db_session.commit()

    resp = await client.get("/api/v1/stocks/WIPRO/indicators")

    assert resp.status_code == 200
    body = resp.json()
    assert body["has_data"] is False
    assert body["sma_20"] is None


async def test_indicators_computes_values_with_enough_bars(app_client, db_session):
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [StockMasterRecord(symbol="TCS", isin=None, name="TCS", series="EQ", listing_date=None, face_value=None)]
    )
    await db_session.commit()

    price_repo = SqlAlchemyHistoricalPriceRepository(db_session)
    today = date.today()
    records = [
        BhavcopyRecord(
            symbol="TCS",
            trade_date=today - timedelta(days=(60 - i)),
            open=Decimal(str(100 + i)),
            high=Decimal(str(101 + i)),
            low=Decimal(str(99 + i)),
            close=Decimal(str(100 + i)),
            volume=1000 + i,
        )
        for i in range(60)
    ]
    await price_repo.bulk_upsert_bars(records)

    client, _ = app_client
    resp = await client.get("/api/v1/stocks/TCS/indicators")

    assert resp.status_code == 200
    body = resp.json()
    assert body["has_data"] is True
    assert body["sma_20"] is not None
    assert body["rsi_14"] is not None
    assert body["pivot_points"] is not None
