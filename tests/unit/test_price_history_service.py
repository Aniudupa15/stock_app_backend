from datetime import date, timedelta
from decimal import Decimal

import pytest

from app.core.exceptions import StockNotFoundError
from app.domain.entities import BhavcopyRecord
from app.services.price_history_service import PriceHistoryService
from tests.conftest import FakeHistoricalPriceRepository, FakeStockDataProvider, FakeStockRepository


@pytest.fixture
def bhavcopy_record() -> BhavcopyRecord:
    return BhavcopyRecord(
        symbol="RELIANCE",
        trade_date=date(2026, 7, 17),
        open=Decimal("2490.00"),
        high=Decimal("2510.00"),
        low=Decimal("2480.00"),
        close=Decimal("2500.00"),
        volume=1_000_000,
    )


async def test_backfill_date_upserts_provider_records(bhavcopy_record, sample_stock):
    repo = FakeHistoricalPriceRepository()
    provider = FakeStockDataProvider(daily_bars=[bhavcopy_record])
    service = PriceHistoryService(repo, provider, FakeStockRepository([sample_stock]))

    upserted = await service.backfill_date(date(2026, 7, 17))

    assert upserted == 1
    assert repo.upserted == [bhavcopy_record]


async def test_backfill_date_returns_zero_for_holiday_with_no_bars(sample_stock):
    repo = FakeHistoricalPriceRepository()
    provider = FakeStockDataProvider(daily_bars=[])
    service = PriceHistoryService(repo, provider, FakeStockRepository([sample_stock]))

    upserted = await service.backfill_date(date(2026, 7, 18))

    assert upserted == 0
    assert repo.upserted == []


async def test_get_history_returns_bars_in_range(bhavcopy_record, sample_stock):
    repo = FakeHistoricalPriceRepository()
    service = PriceHistoryService(repo, FakeStockDataProvider(), FakeStockRepository([sample_stock]))
    await repo.bulk_upsert_bars([bhavcopy_record])

    result = await service.get_history("RELIANCE", "1M")

    assert result.symbol == "RELIANCE"
    assert result.range == "1M"
    assert len(result.bars) == 1
    assert result.bars[0].close == Decimal("2500.00")


async def test_get_history_raises_when_stock_unknown():
    service = PriceHistoryService(FakeHistoricalPriceRepository(), FakeStockDataProvider(), FakeStockRepository())

    with pytest.raises(StockNotFoundError):
        await service.get_history("DOESNOTEXIST", "1M")


async def test_get_history_excludes_bars_outside_range(sample_stock):
    old_record = BhavcopyRecord(
        symbol="RELIANCE",
        trade_date=date.today() - timedelta(days=1000),
        open=Decimal("1000"),
        high=Decimal("1010"),
        low=Decimal("990"),
        close=Decimal("1005"),
        volume=500,
    )
    repo = FakeHistoricalPriceRepository()
    await repo.bulk_upsert_bars([old_record])
    service = PriceHistoryService(repo, FakeStockDataProvider(), FakeStockRepository([sample_stock]))

    result = await service.get_history("RELIANCE", "1M")

    assert result.bars == []


async def test_get_history_unknown_range_falls_back_to_default(bhavcopy_record, sample_stock):
    repo = FakeHistoricalPriceRepository()
    await repo.bulk_upsert_bars([bhavcopy_record])
    service = PriceHistoryService(repo, FakeStockDataProvider(), FakeStockRepository([sample_stock]))

    result = await service.get_history("RELIANCE", "NOT_A_RANGE")

    assert result.range == "1Y"
