from decimal import Decimal

from app.domain.entities import MarketMover
from app.services.market_mover_service import MarketMoverService
from tests.conftest import FakeMarketMoverRepository


def _mover(symbol: str, change_percent: Decimal | None = Decimal("5.00")) -> MarketMover:
    return MarketMover(
        symbol=symbol,
        name=f"{symbol} Ltd",
        last_price=Decimal("100.00"),
        change=Decimal("5.00"),
        change_percent=change_percent,
        volume=1_000,
    )


async def test_get_gainers_maps_period_to_lookback_sessions():
    repo = FakeMarketMoverRepository(top_movers={"gainers": [_mover("A")]})
    service = MarketMoverService(repo)

    result = await service.get_gainers("1M", 10)

    assert result[0].symbol == "A"
    assert repo.calls == [("get_top_movers", "gainers", 21, 10)]


async def test_get_losers_maps_period_to_lookback_sessions():
    repo = FakeMarketMoverRepository(top_movers={"losers": [_mover("B")]})
    service = MarketMoverService(repo)

    result = await service.get_losers("1Y", 5)

    assert result[0].symbol == "B"
    assert repo.calls == [("get_top_movers", "losers", 252, 5)]


async def test_unknown_period_falls_back_to_default():
    repo = FakeMarketMoverRepository(top_movers={"gainers": [_mover("C")]})
    service = MarketMoverService(repo)

    await service.get_gainers("BOGUS", 10)

    assert repo.calls == [("get_top_movers", "gainers", 1, 10)]


async def test_get_most_active_delegates_to_repository():
    repo = FakeMarketMoverRepository(most_active=[_mover("D")])
    service = MarketMoverService(repo)

    result = await service.get_most_active(15)

    assert result[0].symbol == "D"
    assert repo.calls == [("get_most_active", 15)]


async def test_get_52_week_high_and_low_delegate_with_direction():
    repo = FakeMarketMoverRepository(extremes={"high": [_mover("E")], "low": [_mover("F")]})
    service = MarketMoverService(repo)

    high = await service.get_52_week_high(10)
    low = await service.get_52_week_low(10)

    assert high[0].symbol == "E"
    assert low[0].symbol == "F"
    assert repo.calls == [("get_52_week_extremes", "high", 10), ("get_52_week_extremes", "low", 10)]


async def test_get_heatmap_buckets_by_change_percent():
    movers = [
        _mover("A", Decimal("5")),  # STRONG_GAIN
        _mover("B", Decimal("1")),  # GAIN
        _mover("C", Decimal("0")),  # FLAT
        _mover("D", Decimal("-1")),  # LOSS
        _mover("E", Decimal("-5")),  # STRONG_LOSS
        _mover("F", None),  # UNKNOWN
    ]
    repo = FakeMarketMoverRepository(most_active=movers)
    service = MarketMoverService(repo)

    heatmap = await service.get_heatmap(100)

    buckets = {tile.symbol: tile.bucket for tile in heatmap.tiles}
    assert buckets == {
        "A": "STRONG_GAIN",
        "B": "GAIN",
        "C": "FLAT",
        "D": "LOSS",
        "E": "STRONG_LOSS",
        "F": "UNKNOWN",
    }
    assert len(heatmap.notes) == 1
    assert repo.calls == [("get_most_active", 100)]


async def test_get_heatmap_boundary_at_exactly_3_percent_is_strong():
    repo = FakeMarketMoverRepository(most_active=[_mover("G", Decimal("3")), _mover("H", Decimal("-3"))])
    service = MarketMoverService(repo)

    heatmap = await service.get_heatmap(100)

    buckets = {tile.symbol: tile.bucket for tile in heatmap.tiles}
    assert buckets == {"G": "STRONG_GAIN", "H": "STRONG_LOSS"}
