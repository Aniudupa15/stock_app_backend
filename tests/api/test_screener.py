from datetime import date
from decimal import Decimal

from app.domain.entities import StockIndicatorSnapshot, StockMasterRecord
from app.repositories.screener_repository import SqlAlchemyScreenerRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository


async def _seed_snapshot(db_session, symbol: str, rsi_14: Decimal) -> None:
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [
            StockMasterRecord(
                symbol=symbol, isin=None, name=f"{symbol} Ltd", series="EQ", listing_date=None, face_value=None
            )
        ]
    )
    await db_session.commit()

    screener_repo = SqlAlchemyScreenerRepository(db_session)
    await screener_repo.bulk_upsert(
        [
            StockIndicatorSnapshot(
                symbol=symbol,
                name="",
                as_of=date(2026, 7, 22),
                close=Decimal("100"),
                volume=1000,
                rsi_14=rsi_14,
                sma_50=None,
                sma_200=None,
            )
        ]
    )


async def test_screener_endpoint_filters_by_rsi(app_client, db_session):
    client, _ = app_client
    await _seed_snapshot(db_session, "OVERSOLD", Decimal("20"))
    await _seed_snapshot(db_session, "OVERBOUGHT", Decimal("80"))

    resp = await client.post("/api/v1/screener", json={"rsi_below": "30"})

    assert resp.status_code == 200
    body = resp.json()
    assert [r["symbol"] for r in body] == ["OVERSOLD"]


async def test_screener_endpoint_with_no_filters_returns_all(app_client, db_session):
    client, _ = app_client
    await _seed_snapshot(db_session, "A", Decimal("50"))
    await _seed_snapshot(db_session, "B", Decimal("60"))

    resp = await client.post("/api/v1/screener", json={})

    assert resp.status_code == 200
    assert {r["symbol"] for r in resp.json()} == {"A", "B"}
