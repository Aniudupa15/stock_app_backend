from datetime import date

from app.domain.entities import FinancialResultRecord, StockMasterRecord
from app.repositories.financial_result_repository import SqlAlchemyFinancialResultRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository


async def test_long_term_signal_unknown_symbol_returns_404(app_client):
    client, _ = app_client
    resp = await client.get("/api/v1/stocks/DOESNOTEXIST/long-term-signal")
    assert resp.status_code == 404


async def test_long_term_signal_has_data_false_with_no_financial_results(app_client, db_session):
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [
            StockMasterRecord(
                symbol="HDFCBANK", isin=None, name="HDFC Bank", series="EQ", listing_date=None, face_value=None
            )
        ]
    )
    await db_session.commit()

    client, _ = app_client
    resp = await client.get("/api/v1/stocks/HDFCBANK/long-term-signal")

    assert resp.status_code == 200
    body = resp.json()
    assert body["has_data"] is False
    assert body["signal"] == "HOLD"
    assert "not investment advice" in body["disclaimer"].lower()


async def test_long_term_signal_buy_with_strong_growth(app_client, db_session):
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [
            StockMasterRecord(
                symbol="RELIANCE", isin=None, name="Reliance", series="EQ", listing_date=None, face_value=None
            )
        ]
    )
    await db_session.commit()

    financial_repo = SqlAlchemyFinancialResultRepository(db_session)
    await financial_repo.bulk_upsert(
        [
            FinancialResultRecord(
                symbol="RELIANCE",
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
                consolidated=False,
                revenue=1000,
                profit=150,
                eps_basic=5.0,
                eps_diluted=5.0,
            ),
            FinancialResultRecord(
                symbol="RELIANCE",
                period_start=date(2023, 1, 1),
                period_end=date(2023, 12, 31),
                consolidated=False,
                revenue=800,
                profit=100,
                eps_basic=4.0,
                eps_diluted=4.0,
            ),
        ]
    )

    client, _ = app_client
    resp = await client.get("/api/v1/stocks/RELIANCE/long-term-signal")

    assert resp.status_code == 200
    body = resp.json()
    assert body["has_data"] is True
    assert body["signal"] == "BUY"
    assert len(body["strengths"]) > 0
