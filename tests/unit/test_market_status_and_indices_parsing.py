from decimal import Decimal

import httpx
import pytest
import respx

from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError
from app.domain.entities import IndexQuote, MarketStatus
from app.providers.nse.client import NseClient
from app.providers.nse.nse_provider import NseStockDataProvider

HOMEPAGE = "https://www.nseindia.com/"
MARKET_STATUS_URL = "https://www.nseindia.com/api/marketStatus"
ALL_INDICES_URL = "https://www.nseindia.com/api/allIndices"


@pytest.fixture
def settings() -> Settings:
    return Settings(DATABASE_URL="postgresql+asyncpg://test:test@localhost/test", NSE_RATE_LIMIT_PER_SECOND=1000)


def _mock_homepage():
    respx.get(HOMEPAGE).mock(return_value=httpx.Response(200, headers={"set-cookie": "nsit=abc; Path=/"}))


@respx.mock
async def test_fetch_market_status_parses_real_shaped_response(settings):
    _mock_homepage()
    respx.get(MARKET_STATUS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "marketState": [
                    {"market": "Capital Market", "marketStatus": "Open", "tradeDate": "22-Jul-2026 15:30:00"},
                    {"market": "", "marketStatus": "Closed", "tradeDate": "22-Jul-2026"},  # missing market, skipped
                ]
            },
        )
    )

    client = NseClient(settings)
    try:
        statuses = await NseStockDataProvider(client).fetch_market_status()
    finally:
        await client.aclose()

    assert statuses == [MarketStatus(market="Capital Market", status="Open", as_of="22-Jul-2026 15:30:00")]


@respx.mock
async def test_fetch_market_status_unexpected_shape_raises_provider_unavailable(settings):
    _mock_homepage()
    respx.get(MARKET_STATUS_URL).mock(return_value=httpx.Response(200, json={"unexpected": "shape"}))

    client = NseClient(settings)
    try:
        with pytest.raises(ProviderUnavailableError):
            await NseStockDataProvider(client).fetch_market_status()
    finally:
        await client.aclose()


@respx.mock
async def test_fetch_indices_parses_real_shaped_response(settings):
    _mock_homepage()
    respx.get(ALL_INDICES_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"indexSymbol": "NIFTY 50", "last": 25000.5, "variation": 120.3, "percentChange": 0.48},
                    {"indexSymbol": "", "last": 100},  # missing name, skipped
                ]
            },
        )
    )

    client = NseClient(settings)
    try:
        indices = await NseStockDataProvider(client).fetch_indices()
    finally:
        await client.aclose()

    assert indices == [
        IndexQuote(
            index_name="NIFTY 50",
            last_price=Decimal("25000.5"),
            change=Decimal("120.3"),
            change_percent=Decimal("0.48"),
        )
    ]


@respx.mock
async def test_fetch_indices_unexpected_shape_raises_provider_unavailable(settings):
    _mock_homepage()
    respx.get(ALL_INDICES_URL).mock(return_value=httpx.Response(200, json=[]))

    client = NseClient(settings)
    try:
        with pytest.raises(ProviderUnavailableError):
            await NseStockDataProvider(client).fetch_indices()
    finally:
        await client.aclose()
