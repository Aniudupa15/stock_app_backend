import pytest

from app.core.exceptions import ProviderUnavailableError, StockNotFoundError
from app.services.stock_service import StockService
from tests.conftest import FakeCache, FakeStockDataProvider, FakeStockRepository


@pytest.fixture
def service(sample_stock, sample_quote, settings) -> tuple[StockService, FakeStockRepository, FakeStockDataProvider, FakeCache]:
    repo = FakeStockRepository([sample_stock])
    provider = FakeStockDataProvider(quotes={"RELIANCE": sample_quote})
    cache = FakeCache()
    return StockService(repo, provider, cache, settings), repo, provider, cache


async def test_search_returns_matches(service):
    svc, *_ = service
    results = await svc.search("reli", limit=10)
    assert len(results) == 1
    assert results[0].symbol == "RELIANCE"


async def test_search_uses_cache_on_second_call(service):
    svc, repo, _, cache = service
    await svc.search("reli", limit=10)
    # Mutate the repo's underlying data - if the cache is doing its job, the
    # second call must still return the original cached result.
    repo.stocks.clear()
    results = await svc.search("reli", limit=10)
    assert len(results) == 1
    assert results[0].symbol == "RELIANCE"


async def test_get_detail_not_found_raises_domain_error(service):
    svc, *_ = service
    with pytest.raises(StockNotFoundError):
        await svc.get_detail("DOESNOTEXIST")


async def test_get_detail_includes_live_quote(service):
    svc, *_ = service
    detail = await svc.get_detail("RELIANCE")
    assert detail.symbol == "RELIANCE"
    assert detail.quote is not None
    assert detail.quote.last_price == pytest.approx(2500.00)
    assert detail.quote_unavailable_reason is None


async def test_get_detail_degrades_gracefully_when_provider_fails(service):
    svc, _, provider, _ = service
    provider.fail_with = ProviderUnavailableError("NSE", "circuit open")

    detail = await svc.get_detail("RELIANCE")

    assert detail.symbol == "RELIANCE"
    assert detail.quote is None
    assert detail.quote_unavailable_reason is not None
