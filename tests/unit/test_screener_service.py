from datetime import date
from decimal import Decimal

from app.domain.entities import StockIndicatorSnapshot
from app.schemas.screener import ScreenerRequest
from app.services.screener_service import ScreenerService
from tests.conftest import FakeScreenerRepository


def _snapshot(symbol: str, rsi_14=None, close=Decimal("100"), sma_50=None, volume=1000) -> StockIndicatorSnapshot:
    return StockIndicatorSnapshot(
        symbol=symbol,
        name=f"{symbol} Ltd",
        as_of=date(2026, 7, 22),
        close=close,
        volume=volume,
        rsi_14=rsi_14,
        sma_50=sma_50,
        sma_200=None,
    )


async def test_screen_filters_by_rsi_below():
    repo = FakeScreenerRepository([_snapshot("A", rsi_14=Decimal("20")), _snapshot("B", rsi_14=Decimal("80"))])
    service = ScreenerService(repo)

    result = await service.screen(ScreenerRequest(rsi_below=Decimal("30")))

    assert [r.symbol for r in result] == ["A"]


async def test_screen_filters_by_rsi_above():
    repo = FakeScreenerRepository([_snapshot("A", rsi_14=Decimal("20")), _snapshot("B", rsi_14=Decimal("80"))])
    service = ScreenerService(repo)

    result = await service.screen(ScreenerRequest(rsi_above=Decimal("70")))

    assert [r.symbol for r in result] == ["B"]


async def test_screen_filters_by_price_range():
    repo = FakeScreenerRepository([_snapshot("A", close=Decimal("50")), _snapshot("B", close=Decimal("500"))])
    service = ScreenerService(repo)

    result = await service.screen(ScreenerRequest(price_min=Decimal("100"), price_max=Decimal("1000")))

    assert [r.symbol for r in result] == ["B"]


async def test_screen_filters_above_sma_50():
    repo = FakeScreenerRepository(
        [
            _snapshot("ABOVE", close=Decimal("110"), sma_50=Decimal("100")),
            _snapshot("BELOW", close=Decimal("90"), sma_50=Decimal("100")),
        ]
    )
    service = ScreenerService(repo)

    above = await service.screen(ScreenerRequest(above_sma_50=True))
    below = await service.screen(ScreenerRequest(above_sma_50=False))

    assert [r.symbol for r in above] == ["ABOVE"]
    assert [r.symbol for r in below] == ["BELOW"]


async def test_screen_filters_by_min_volume():
    repo = FakeScreenerRepository([_snapshot("LOW", volume=100), _snapshot("HIGH", volume=100_000)])
    service = ScreenerService(repo)

    result = await service.screen(ScreenerRequest(min_volume=10_000))

    assert [r.symbol for r in result] == ["HIGH"]


async def test_screen_with_no_filters_returns_everything():
    repo = FakeScreenerRepository([_snapshot("A"), _snapshot("B")])
    service = ScreenerService(repo)

    result = await service.screen(ScreenerRequest())

    assert {r.symbol for r in result} == {"A", "B"}
