from datetime import date
from decimal import Decimal

from app.domain.entities import BhavcopyRecord, StockMasterRecord
from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository


async def _seed_bars(db_session, symbol: str, closes: dict[date, tuple[Decimal, int]]) -> None:
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
            BhavcopyRecord(symbol=symbol, trade_date=d, open=close, high=close, low=close, close=close, volume=volume)
            for d, (close, volume) in closes.items()
        ]
    )


async def test_gainers_and_losers_endpoints_return_ranked_movers(app_client, db_session):
    client, _ = app_client
    await _seed_bars(
        db_session, "UP", {date(2026, 7, 20): (Decimal("100"), 500), date(2026, 7, 21): (Decimal("110"), 500)}
    )
    await _seed_bars(
        db_session, "DOWN", {date(2026, 7, 20): (Decimal("100"), 500), date(2026, 7, 21): (Decimal("90"), 500)}
    )

    gainers_resp = await client.get("/api/v1/market/gainers", params={"period": "1D"})
    losers_resp = await client.get("/api/v1/market/losers", params={"period": "1D"})

    assert gainers_resp.status_code == 200
    assert gainers_resp.json()[0]["symbol"] == "UP"
    assert losers_resp.status_code == 200
    assert losers_resp.json()[0]["symbol"] == "DOWN"


async def test_most_active_endpoint_ranks_by_volume(app_client, db_session):
    client, _ = app_client
    await _seed_bars(db_session, "QUIET", {date(2026, 7, 21): (Decimal("50"), 100)})
    await _seed_bars(db_session, "BUSY", {date(2026, 7, 21): (Decimal("50"), 9_999)})

    resp = await client.get("/api/v1/market/most-active")

    assert resp.status_code == 200
    assert resp.json()[0]["symbol"] == "BUSY"


async def test_52_week_high_and_low_endpoints(app_client, db_session):
    client, _ = app_client
    await _seed_bars(
        db_session,
        "NEWHIGH",
        {date(2026, 7, 19): (Decimal("90"), 100), date(2026, 7, 21): (Decimal("100"), 100)},
    )
    await _seed_bars(
        db_session,
        "NEWLOW",
        {date(2026, 7, 19): (Decimal("100"), 100), date(2026, 7, 21): (Decimal("90"), 100)},
    )

    high_resp = await client.get("/api/v1/market/52-week-high")
    low_resp = await client.get("/api/v1/market/52-week-low")

    assert high_resp.status_code == 200
    assert "NEWHIGH" in [m["symbol"] for m in high_resp.json()]
    assert low_resp.status_code == 200
    assert "NEWLOW" in [m["symbol"] for m in low_resp.json()]


async def test_heatmap_endpoint_returns_bucketed_tiles(app_client, db_session):
    client, _ = app_client
    await _seed_bars(
        db_session, "UP", {date(2026, 7, 20): (Decimal("100"), 500), date(2026, 7, 21): (Decimal("110"), 500)}
    )
    await _seed_bars(
        db_session, "DOWN", {date(2026, 7, 20): (Decimal("100"), 500), date(2026, 7, 21): (Decimal("90"), 500)}
    )

    resp = await client.get("/api/v1/market/heatmap")

    assert resp.status_code == 200
    body = resp.json()
    tiles_by_symbol = {t["symbol"]: t for t in body["tiles"]}
    assert tiles_by_symbol["UP"]["bucket"] == "STRONG_GAIN"
    assert tiles_by_symbol["DOWN"]["bucket"] == "STRONG_LOSS"
    assert len(body["notes"]) == 1
