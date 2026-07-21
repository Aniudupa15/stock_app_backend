from datetime import date, timedelta
from decimal import Decimal

import pytest

from app.core.exceptions import StockNotFoundError
from app.domain.entities import OhlcvBar
from app.services.indicator_service import IndicatorService
from tests.conftest import FakeHistoricalPriceRepository, FakeStockRepository


def _make_bars(n: int, start_price: float = 100.0) -> list[OhlcvBar]:
    today = date.today()
    bars = []
    price = start_price
    for i in range(n):
        price += 1.0
        trade_date = today - timedelta(days=(n - i))
        bars.append(
            OhlcvBar(
                trade_date=trade_date,
                open=Decimal(str(price - 0.5)),
                high=Decimal(str(price + 1)),
                low=Decimal(str(price - 1)),
                close=Decimal(str(price)),
                volume=10_000 + i,
            )
        )
    return bars


async def test_get_indicators_raises_when_stock_unknown():
    stock_repo = FakeStockRepository()
    price_repo = FakeHistoricalPriceRepository()
    service = IndicatorService(stock_repo, price_repo)

    with pytest.raises(StockNotFoundError):
        await service.get_indicators("DOESNOTEXIST")


async def test_get_indicators_no_data_returns_has_data_false(sample_stock):
    stock_repo = FakeStockRepository([sample_stock])
    price_repo = FakeHistoricalPriceRepository()
    service = IndicatorService(stock_repo, price_repo)

    result = await service.get_indicators("RELIANCE")

    assert result.symbol == "RELIANCE"
    assert result.has_data is False
    assert result.sma_20 is None


async def test_get_indicators_computes_real_values_with_enough_bars(sample_stock):
    stock_repo = FakeStockRepository([sample_stock])
    bars = _make_bars(60)
    price_repo = FakeHistoricalPriceRepository(bars={"RELIANCE": bars})
    service = IndicatorService(stock_repo, price_repo)

    result = await service.get_indicators("RELIANCE")

    assert result.has_data is True
    assert result.sma_20 is not None
    assert result.rsi_14 is not None
    # Monotonically rising closes -> RSI should be strongly bullish (>70)
    assert result.rsi_14 > 70
    assert result.pivot_points is not None
    assert result.point_of_control is not None
