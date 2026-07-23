from datetime import date
from decimal import Decimal

from app.core.auth import DEFAULT_USER_ID
from app.domain.entities import BhavcopyRecord, StockMasterRecord
from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from app.repositories.notification_repository import SqlAlchemyNotificationRepository
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


async def test_chat_unknown_question_returns_help_text(app_client):
    client, _ = app_client
    resp = await client.post("/api/v1/chat", json={"message": "what's the weather like"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "unknown"


async def test_chat_portfolio_question_with_no_portfolios(app_client):
    client, _ = app_client
    resp = await client.post("/api/v1/chat", json={"message": "how is my portfolio doing"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "portfolio_summary"
    assert "don't have any portfolios" in body["answer"]


async def test_chat_alerts_question_with_unread_notification(app_client, db_session):
    client, _ = app_client
    repo = SqlAlchemyNotificationRepository(db_session)
    await repo.create(DEFAULT_USER_ID, None, "Price alert", "RELIANCE crossed target")

    resp = await client.post("/api/v1/chat", json={"message": "any new alerts"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "alerts_summary"
    assert "Price alert" in body["answer"]


async def test_chat_indicator_question_matches_a_real_symbol(app_client, db_session):
    client, _ = app_client
    await _seed_stock_with_price(db_session, "RELIANCE")

    resp = await client.post("/api/v1/chat", json={"message": "what's the RSI for RELIANCE"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "indicator_summary"
    assert "RELIANCE" in body["answer"]


async def test_chat_rejects_empty_message(app_client):
    client, _ = app_client
    resp = await client.post("/api/v1/chat", json={"message": ""})
    assert resp.status_code == 422
