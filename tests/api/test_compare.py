from dataclasses import replace
from datetime import date
from decimal import Decimal

from app.domain.entities import BhavcopyRecord, StockMasterRecord
from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository


def _register_fake_quote(fake_provider, sample_quote, symbol: str) -> None:
    # app_client's fake provider only pre-registers a "RELIANCE" quote -
    # compare needs 2+ real symbols, so register fakes for whichever ones
    # this test actually uses (content doesn't matter, just presence).
    fake_provider.quotes[symbol] = replace(sample_quote, symbol=symbol)


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


async def test_compare_returns_entry_per_symbol(app_client, db_session, sample_quote):
    client, fake_provider = app_client
    await _seed_stock_with_price(db_session, "TCS")
    await _seed_stock_with_price(db_session, "INFY")
    _register_fake_quote(fake_provider, sample_quote, "TCS")
    _register_fake_quote(fake_provider, sample_quote, "INFY")

    resp = await client.get("/api/v1/stocks/compare", params={"symbols": "TCS,INFY"})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["entries"]) == 2
    assert {e["detail"]["symbol"] for e in body["entries"]} == {"TCS", "INFY"}


async def test_compare_rejects_single_symbol(app_client):
    client, _ = app_client
    resp = await client.get("/api/v1/stocks/compare", params={"symbols": "TCS"})
    assert resp.status_code == 422


async def test_compare_rejects_too_many_symbols(app_client):
    client, _ = app_client
    resp = await client.get("/api/v1/stocks/compare", params={"symbols": "A,B,C,D,E,F"})
    assert resp.status_code == 422


async def test_compare_unknown_symbol_returns_404(app_client, db_session, sample_quote):
    client, fake_provider = app_client
    await _seed_stock_with_price(db_session, "TCS")
    _register_fake_quote(fake_provider, sample_quote, "TCS")

    resp = await client.get("/api/v1/stocks/compare", params={"symbols": "TCS,DOESNOTEXIST"})

    assert resp.status_code == 404
