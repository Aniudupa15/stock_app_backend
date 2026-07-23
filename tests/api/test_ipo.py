from datetime import date

from app.domain.entities import IpoFiling
from app.repositories.ipo_repository import SqlAlchemyIpoRepository


def _filing(symbol: str, status: str = "ACTIVE") -> IpoFiling:
    return IpoFiling(
        symbol=symbol,
        company_name=f"{symbol} Ltd",
        status=status,
        price_range="Rs.100 to Rs.110",
        issue_size="1,00,00,000",
        issue_start_date=date(2026, 7, 20),
        issue_end_date=date(2026, 7, 23),
        listing_date=None,
        series="EQ",
    )


async def test_list_ipos_returns_seeded_filings(app_client, db_session):
    client, _ = app_client
    repo = SqlAlchemyIpoRepository(db_session)
    await repo.bulk_upsert([_filing("NEWCO", status="ACTIVE"), _filing("OLDCO", status="LISTED")])

    resp = await client.get("/api/v1/ipo")

    assert resp.status_code == 200
    assert {row["symbol"] for row in resp.json()} == {"NEWCO", "OLDCO"}


async def test_list_ipos_filters_by_status(app_client, db_session):
    client, _ = app_client
    repo = SqlAlchemyIpoRepository(db_session)
    await repo.bulk_upsert([_filing("NEWCO", status="ACTIVE"), _filing("OLDCO", status="LISTED")])

    resp = await client.get("/api/v1/ipo", params={"status": "LISTED"})

    assert resp.status_code == 200
    assert [row["symbol"] for row in resp.json()] == ["OLDCO"]


async def test_list_ipos_with_no_data_returns_empty_list(app_client):
    client, _ = app_client

    resp = await client.get("/api/v1/ipo")

    assert resp.status_code == 200
    assert resp.json() == []
