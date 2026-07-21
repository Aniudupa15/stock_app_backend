from datetime import date

from app.domain.entities import FinancialResultRecord, StockMasterRecord
from app.repositories.financial_result_repository import SqlAlchemyFinancialResultRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository


async def test_fundamentals_unknown_symbol_returns_404(app_client):
    client, _ = app_client
    resp = await client.get("/api/v1/stocks/DOESNOTEXIST/fundamentals")
    assert resp.status_code == 404


async def test_fundamentals_has_data_false_with_no_financial_results(app_client, db_session):
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [StockMasterRecord(symbol="INFY", isin=None, name="Infosys", series="EQ", listing_date=None, face_value=None)]
    )
    await db_session.commit()

    client, _ = app_client
    resp = await client.get("/api/v1/stocks/INFY/fundamentals")

    assert resp.status_code == 200
    body = resp.json()
    assert body["has_data"] is False
    # Always present, always null - not silently omitted.
    assert body["book_value"] is None
    assert body["roe"] is None
    assert body["roce"] is None
    assert body["debt_to_equity"] is None


async def test_fundamentals_returns_real_data_when_quarters_stored(app_client, db_session):
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [StockMasterRecord(symbol="RELIANCE", isin=None, name="Reliance", series="EQ", listing_date=None, face_value=None)]
    )
    await db_session.commit()

    financial_repo = SqlAlchemyFinancialResultRepository(db_session)
    await financial_repo.bulk_upsert(
        [
            FinancialResultRecord(
                symbol="RELIANCE",
                period_start=date(2024, 10, 1),
                period_end=date(2024, 12, 31),
                consolidated=False,
                revenue=1282600000000,
                profit=87210000000,
                eps_basic=6.44,
                eps_diluted=6.44,
            )
        ]
    )

    client, _ = app_client
    resp = await client.get("/api/v1/stocks/RELIANCE/fundamentals")

    assert resp.status_code == 200
    body = resp.json()
    assert body["has_data"] is True
    assert body["latest_period_end"] == "2024-12-31"
