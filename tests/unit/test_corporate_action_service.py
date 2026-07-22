from datetime import date

import pytest

from app.core.exceptions import StockNotFoundError
from app.domain.entities import CorporateAction
from app.services.corporate_action_service import CorporateActionService
from tests.conftest import FakeCorporateActionRepository, FakeStockDataProvider, FakeStockRepository


@pytest.fixture
def corporate_action() -> CorporateAction:
    return CorporateAction(
        symbol="RELIANCE",
        purpose="Dividend - Rs 10 Per Share",
        face_value=None,
        ex_date=date(2026, 8, 1),
        record_date=date(2026, 8, 1),
        book_closure_start=None,
        book_closure_end=None,
    )


async def test_sync_upserts_provider_records(corporate_action):
    repo = FakeCorporateActionRepository()
    provider = FakeStockDataProvider(corporate_actions=[corporate_action])
    service = CorporateActionService(repo, provider, FakeStockRepository())

    upserted = await service.sync(date(2026, 7, 1), date(2026, 9, 1))

    assert upserted == 1
    assert repo.actions_by_symbol["RELIANCE"] == [corporate_action]


async def test_get_for_symbol_raises_when_stock_unknown():
    repo = FakeCorporateActionRepository()
    provider = FakeStockDataProvider()
    service = CorporateActionService(repo, provider, FakeStockRepository())

    with pytest.raises(StockNotFoundError):
        await service.get_for_symbol("DOESNOTEXIST")


async def test_get_for_symbol_returns_stored_actions(sample_stock, corporate_action):
    repo = FakeCorporateActionRepository(actions={"RELIANCE": [corporate_action]})
    provider = FakeStockDataProvider()
    service = CorporateActionService(repo, provider, FakeStockRepository([sample_stock]))

    results = await service.get_for_symbol("RELIANCE")

    assert len(results) == 1
    assert results[0].purpose == "Dividend - Rs 10 Per Share"
