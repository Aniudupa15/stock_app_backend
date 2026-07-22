from datetime import date, timedelta
from decimal import Decimal

from app.domain.entities import BhavcopyRecord, StockMasterRecord
from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository


async def test_intraday_signal_unknown_symbol_returns_404(app_client):
    client, _ = app_client
    resp = await client.get("/api/v1/stocks/DOESNOTEXIST/intraday-signal")
    assert resp.status_code == 404


async def test_intraday_signal_has_data_false_with_too_few_bars(app_client, db_session):
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [StockMasterRecord(symbol="WIPRO", isin=None, name="Wipro", series="EQ", listing_date=None, face_value=None)]
    )
    await db_session.commit()

    client, _ = app_client
    resp = await client.get("/api/v1/stocks/WIPRO/intraday-signal")

    assert resp.status_code == 200
    body = resp.json()
    assert body["has_data"] is False
    assert body["signal"] == "HOLD"
    assert "not investment advice" in body["disclaimer"].lower()


async def test_intraday_signal_returns_real_computed_signal_with_enough_bars(app_client, db_session):
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
            trade_date=today - timedelta(days=(80 - i)),
            open=Decimal(str(100 + i * 1.5)),
            high=Decimal(str(101 + i * 1.5)),
            low=Decimal(str(99 + i * 1.5)),
            close=Decimal(str(100 + i * 1.5)),
            volume=1000 + i,
        )
        for i in range(80)
    ]
    await price_repo.bulk_upsert_bars(records)

    client, _ = app_client
    resp = await client.get("/api/v1/stocks/TCS/intraday-signal")

    assert resp.status_code == 200
    body = resp.json()
    assert body["has_data"] is True
    assert body["signal"] in ("BUY", "SELL", "HOLD")
    assert isinstance(body["reasoning"], list)
    assert len(body["reasoning"]) > 0
