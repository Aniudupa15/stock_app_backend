import uuid
from datetime import date
from decimal import Decimal

from app.domain.entities import BhavcopyRecord, StockMasterRecord
from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository


async def _seed_stock_with_price(db_session, symbol: str, close: str) -> None:
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
                open=Decimal(close),
                high=Decimal(close),
                low=Decimal(close),
                close=Decimal(close),
                volume=1000,
            )
        ]
    )


async def test_create_add_transaction_and_view_holdings(app_client, db_session):
    client, _ = app_client
    await _seed_stock_with_price(db_session, "RELIANCE", "150")

    create_resp = await client.post("/api/v1/portfolios", json={"name": "Long Term"})
    assert create_resp.status_code == 201
    portfolio_id = create_resp.json()["id"]

    txn_resp = await client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={
            "symbol": "RELIANCE",
            "transaction_type": "BUY",
            "quantity": "10",
            "price": "100",
            "transaction_date": "2026-01-01",
        },
    )
    assert txn_resp.status_code == 200
    body = txn_resp.json()
    assert body["holdings"][0]["symbol"] == "RELIANCE"
    assert body["holdings"][0]["current_price"] == "150.00"
    assert Decimal(body["holdings"][0]["pnl"]) == Decimal("500.00")

    detail_resp = await client.get(f"/api/v1/portfolios/{portfolio_id}")
    assert detail_resp.status_code == 200
    assert len(detail_resp.json()["holdings"]) == 1

    perf_resp = await client.get(f"/api/v1/portfolios/{portfolio_id}/performance")
    assert perf_resp.status_code == 200
    perf_body = perf_resp.json()
    assert Decimal(perf_body["total_invested"]) == Decimal("1000.00")
    assert Decimal(perf_body["current_value"]) == Decimal("1500.00")


async def test_add_transaction_unknown_symbol_returns_404(app_client):
    client, _ = app_client
    create_resp = await client.post("/api/v1/portfolios", json={"name": "Long Term"})
    portfolio_id = create_resp.json()["id"]

    resp = await client.post(
        f"/api/v1/portfolios/{portfolio_id}/transactions",
        json={
            "symbol": "DOESNOTEXIST",
            "transaction_type": "BUY",
            "quantity": "10",
            "price": "100",
            "transaction_date": "2026-01-01",
        },
    )

    assert resp.status_code == 404


async def test_unknown_portfolio_returns_404(app_client):
    client, _ = app_client
    resp = await client.get(f"/api/v1/portfolios/{uuid.uuid4()}")
    assert resp.status_code == 404
