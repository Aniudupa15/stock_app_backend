from datetime import date

from app.domain.entities import IpoFiling
from app.services.ipo_service import IpoService
from tests.conftest import FakeIpoRepository, FakeStockDataProvider


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


async def test_sync_upserts_filings_from_provider():
    provider = FakeStockDataProvider(ipo_filings=[_filing("NEWCO"), _filing("OTHERCO")])
    repository = FakeIpoRepository()
    service = IpoService(provider, repository)

    upserted = await service.sync()

    assert upserted == 2
    assert set(repository.filings_by_symbol) == {"NEWCO", "OTHERCO"}


async def test_sync_returns_zero_when_provider_has_nothing():
    provider = FakeStockDataProvider(ipo_filings=[])
    repository = FakeIpoRepository()
    service = IpoService(provider, repository)

    upserted = await service.sync()

    assert upserted == 0
    assert repository.filings_by_symbol == {}


async def test_list_returns_schema_objects_from_repository():
    repository = FakeIpoRepository([_filing("NEWCO"), _filing("OLDCO", status="LISTED")])
    service = IpoService(FakeStockDataProvider(), repository)

    result = await service.list(status=None, limit=20, offset=0)

    assert {r.symbol for r in result} == {"NEWCO", "OLDCO"}


async def test_list_filters_by_status():
    repository = FakeIpoRepository([_filing("NEWCO"), _filing("OLDCO", status="LISTED")])
    service = IpoService(FakeStockDataProvider(), repository)

    result = await service.list(status="LISTED", limit=20, offset=0)

    assert [r.symbol for r in result] == ["OLDCO"]
