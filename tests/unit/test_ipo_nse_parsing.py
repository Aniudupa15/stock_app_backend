from datetime import date

import httpx
import pytest
import respx

from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError
from app.domain.entities import IpoFiling
from app.providers.nse.client import NseClient
from app.providers.nse.nse_provider import NseStockDataProvider

IPO_WARMUP_URL = "https://www.nseindia.com/market-data/all-upcoming-issues-ipo"
UPCOMING_URL = "https://www.nseindia.com/api/all-upcoming-issues"
PAST_URL = "https://www.nseindia.com/api/public-past-issues"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        DATABASE_URL="postgresql+asyncpg://test:test@localhost/test",
        NSE_RATE_LIMIT_PER_SECOND=1000,
        NSE_MAX_RETRIES=1,
    )


def _mock_warmup():
    respx.get(IPO_WARMUP_URL).mock(return_value=httpx.Response(200, headers={"set-cookie": "nsit=abc; Path=/"}))


@respx.mock
async def test_fetch_ipo_filings_parses_both_endpoints(settings):
    _mock_warmup()
    respx.get(UPCOMING_URL).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "symbol": "newco",
                    "companyName": "New Co Ltd",
                    "status": "active",
                    "issuePrice": "Rs.120 to Rs.127",
                    "issueSize": "1,00,00,000",
                    "issueStartDate": "20-Jul-2026",
                    "issueEndDate": "23-Jul-2026",
                    "series": "EQ",
                },
                {"symbol": "", "companyName": "Skip Me"},  # missing symbol, skipped
            ],
        )
    )
    respx.get(PAST_URL).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "symbol": "oldco",
                    "company": "Old Co Ltd",
                    "priceRange": "Rs.50 to Rs.55",
                    "ipoStartDate": "01-Jun-2026",
                    "ipoEndDate": "03-Jun-2026",
                    "listingDate": "10-Jun-2026",
                    "securityType": "EQ",
                }
            ],
        )
    )

    client = NseClient(settings)
    try:
        filings = await NseStockDataProvider(client).fetch_ipo_filings()
    finally:
        await client.aclose()

    assert filings == [
        IpoFiling(
            symbol="NEWCO",
            company_name="New Co Ltd",
            status="ACTIVE",
            price_range="Rs.120 to Rs.127",
            issue_size="1,00,00,000",
            issue_start_date=date(2026, 7, 20),
            issue_end_date=date(2026, 7, 23),
            listing_date=None,
            series="EQ",
        ),
        IpoFiling(
            symbol="OLDCO",
            company_name="Old Co Ltd",
            status="LISTED",
            price_range="Rs.50 to Rs.55",
            issue_size=None,
            issue_start_date=date(2026, 6, 1),
            issue_end_date=date(2026, 6, 3),
            listing_date=date(2026, 6, 10),
            series="EQ",
        ),
    ]


@respx.mock
async def test_fetch_ipo_filings_partial_failure_returns_the_other_endpoints_data(settings):
    _mock_warmup()
    respx.get(UPCOMING_URL).mock(return_value=httpx.Response(500))
    respx.get(PAST_URL).mock(
        return_value=httpx.Response(
            200,
            json=[{"symbol": "OLDCO", "company": "Old Co Ltd"}],
        )
    )

    client = NseClient(settings)
    try:
        filings = await NseStockDataProvider(client).fetch_ipo_filings()
    finally:
        await client.aclose()

    assert len(filings) == 1
    assert filings[0].symbol == "OLDCO"
    assert filings[0].status == "LISTED"


@respx.mock
async def test_fetch_ipo_filings_both_endpoints_fail_raises_provider_unavailable(settings):
    _mock_warmup()
    respx.get(UPCOMING_URL).mock(return_value=httpx.Response(500))
    respx.get(PAST_URL).mock(return_value=httpx.Response(500))

    client = NseClient(settings)
    try:
        with pytest.raises(ProviderUnavailableError):
            await NseStockDataProvider(client).fetch_ipo_filings()
    finally:
        await client.aclose()
