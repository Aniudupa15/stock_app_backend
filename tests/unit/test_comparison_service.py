import pytest

from app.core.exceptions import StockNotFoundError
from app.services.comparison_service import ComparisonService
from app.services.fundamentals_service import FundamentalsService
from app.services.indicator_service import IndicatorService
from app.services.stock_service import StockService
from tests.conftest import (
    FakeCache,
    FakeCorporateActionRepository,
    FakeFinancialResultRepository,
    FakeHistoricalPriceRepository,
    FakeStockDataProvider,
    FakeStockRepository,
)


@pytest.fixture
def service(sample_stock, sample_quote, settings) -> ComparisonService:
    stock_repo = FakeStockRepository([sample_stock])
    stock_service = StockService(
        stock_repo, FakeStockDataProvider(quotes={"RELIANCE": sample_quote}), FakeCache(), settings
    )
    indicator_service = IndicatorService(stock_repo, FakeHistoricalPriceRepository())
    fundamentals_service = FundamentalsService(
        stock_repo, FakeFinancialResultRepository(), FakeHistoricalPriceRepository(), FakeCorporateActionRepository()
    )
    return ComparisonService(stock_service, indicator_service, fundamentals_service)


async def test_compare_composes_all_three_services_per_symbol(service):
    result = await service.compare(["RELIANCE"])

    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.detail.symbol == "RELIANCE"
    assert entry.indicators.symbol == "RELIANCE"
    assert entry.fundamentals.has_data is False  # no financial results seeded


async def test_compare_unknown_symbol_raises_stock_not_found(service):
    with pytest.raises(StockNotFoundError):
        await service.compare(["RELIANCE", "DOESNOTEXIST"])
