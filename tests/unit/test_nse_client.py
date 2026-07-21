import httpx
import pytest
import respx

from app.core.config import Settings
from app.providers.nse.circuit_breaker import CircuitOpenError
from app.providers.nse.client import NseClient
from app.providers.nse.exceptions import NseServerError, NseTimeoutError

HOMEPAGE = "https://www.nseindia.com/"
QUOTE_URL = "https://www.nseindia.com/api/quote-equity"


@pytest.fixture
def settings() -> Settings:
    # Overrides the conftest-wide `settings` fixture with NSE-tuned values so
    # rate limiting/backoff don't slow these tests down.
    return Settings(
        DATABASE_URL="postgresql+asyncpg://test:test@localhost/test",
        NSE_MAX_RETRIES=3,
        NSE_CIRCUIT_FAIL_MAX=5,
        NSE_CIRCUIT_RESET_TIMEOUT_SECONDS=60,
        NSE_COOKIE_TTL_SECONDS=300,
        NSE_RATE_LIMIT_PER_SECOND=1000,
    )


def _mock_homepage():
    respx.get(HOMEPAGE).mock(return_value=httpx.Response(200, headers={"set-cookie": "nsit=abc; Path=/"}))


@respx.mock
async def test_bootstraps_session_then_succeeds(settings):
    _mock_homepage()
    respx.get(QUOTE_URL).mock(return_value=httpx.Response(200, json={"priceInfo": {"lastPrice": 100}}))

    client = NseClient(settings)
    try:
        data = await client.get_json("/api/quote-equity", params={"symbol": "TCS"})
    finally:
        await client.aclose()

    assert data["priceInfo"]["lastPrice"] == 100
    assert respx.get(HOMEPAGE).call_count == 1


@respx.mock
async def test_bootstrap_403_is_typed_and_retried(settings):
    """Regression test: a 403 on the homepage bootstrap itself (not the API
    call) must surface as a typed, retryable NseServerError - not an
    unhandled httpx.HTTPStatusError that bypasses retry/circuit-breaker
    logic and the ProviderUnavailableError translation in nse_provider.py.
    """
    route = respx.get(HOMEPAGE)
    route.side_effect = [
        httpx.Response(403),
        httpx.Response(200, headers={"set-cookie": "nsit=abc; Path=/"}),
    ]
    respx.get(QUOTE_URL).mock(return_value=httpx.Response(200, json={"ok": True}))

    client = NseClient(settings)
    try:
        data = await client.get_json("/api/quote-equity", params={"symbol": "TCS"})
    finally:
        await client.aclose()

    assert data == {"ok": True}
    assert route.call_count == 2


@respx.mock
async def test_bootstrap_always_failing_raises_typed_error(settings):
    respx.get(HOMEPAGE).mock(return_value=httpx.Response(403))
    respx.get(QUOTE_URL).mock(return_value=httpx.Response(200, json={"ok": True}))

    client = NseClient(settings)
    try:
        with pytest.raises(NseServerError):
            await client.get_json("/api/quote-equity", params={"symbol": "TCS"})
    finally:
        await client.aclose()


@respx.mock
async def test_403_triggers_rebootstrap_and_retry(settings):
    _mock_homepage()
    route = respx.get(QUOTE_URL)
    route.side_effect = [httpx.Response(403), httpx.Response(200, json={"ok": True})]

    client = NseClient(settings)
    try:
        data = await client.get_json("/api/quote-equity", params={"symbol": "TCS"})
    finally:
        await client.aclose()

    assert data == {"ok": True}
    # initial bootstrap + forced rebootstrap after the 403
    assert respx.get(HOMEPAGE).call_count == 2


@respx.mock
async def test_429_respects_retry_after_then_succeeds(settings):
    _mock_homepage()
    route = respx.get(QUOTE_URL)
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json={"ok": True}),
    ]

    client = NseClient(settings)
    try:
        data = await client.get_json("/api/quote-equity", params={"symbol": "TCS"})
    finally:
        await client.aclose()

    assert data == {"ok": True}


@respx.mock
async def test_404_is_not_retried():
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://test:test@localhost/test",
        NSE_MAX_RETRIES=5,
        NSE_RATE_LIMIT_PER_SECOND=1000,
    )
    _mock_homepage()
    route = respx.get(QUOTE_URL)
    route.mock(return_value=httpx.Response(404))

    client = NseClient(settings)
    try:
        with pytest.raises(Exception):
            await client.get_json("/api/quote-equity", params={"symbol": "NOPE"})
    finally:
        await client.aclose()

    # one attempt only - 404 must not trigger tenacity retries
    assert route.call_count == 1


@respx.mock
async def test_timeout_raises_typed_exception(settings):
    _mock_homepage()
    respx.get(QUOTE_URL).mock(side_effect=httpx.TimeoutException("timed out"))

    client = NseClient(settings)
    try:
        with pytest.raises(NseTimeoutError):
            await client.get_json("/api/quote-equity", params={"symbol": "TCS"})
    finally:
        await client.aclose()


@respx.mock
async def test_circuit_breaker_opens_after_repeated_failures(settings):
    settings = settings.model_copy(update={"NSE_CIRCUIT_FAIL_MAX": 2, "NSE_MAX_RETRIES": 1})
    _mock_homepage()
    respx.get(QUOTE_URL).mock(return_value=httpx.Response(500))

    client = NseClient(settings)
    try:
        for _ in range(2):
            with pytest.raises(NseServerError):
                await client.get_json("/api/quote-equity", params={"symbol": "TCS"})

        with pytest.raises(CircuitOpenError):
            await client.get_json("/api/quote-equity", params={"symbol": "TCS"})
    finally:
        await client.aclose()
