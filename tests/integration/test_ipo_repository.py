from datetime import date

from app.domain.entities import IpoFiling
from app.repositories.ipo_repository import SqlAlchemyIpoRepository
from tests.conftest import requires_docker

pytestmark = requires_docker


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


async def test_bulk_upsert_is_idempotent_per_symbol(db_session):
    repo = SqlAlchemyIpoRepository(db_session)

    first = await repo.bulk_upsert([_filing("NEWCO", status="ACTIVE")])
    second = await repo.bulk_upsert([_filing("NEWCO", status="LISTED")])

    assert first == 1
    assert second == 1

    results = await repo.list_all(status=None, limit=10, offset=0)
    assert len(results) == 1
    assert results[0].status == "LISTED"  # overwritten, not duplicated


async def test_list_all_filters_by_status(db_session):
    repo = SqlAlchemyIpoRepository(db_session)
    await repo.bulk_upsert([_filing("ACTIVEONE", status="ACTIVE"), _filing("LISTEDONE", status="LISTED")])

    results = await repo.list_all(status="LISTED", limit=10, offset=0)

    assert [r.symbol for r in results] == ["LISTEDONE"]


async def test_list_all_respects_limit_and_offset(db_session):
    repo = SqlAlchemyIpoRepository(db_session)
    await repo.bulk_upsert([_filing("A"), _filing("B"), _filing("C")])

    page = await repo.list_all(status=None, limit=1, offset=1)

    assert len(page) == 1
