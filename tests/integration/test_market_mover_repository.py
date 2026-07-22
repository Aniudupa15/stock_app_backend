from datetime import date
from decimal import Decimal

from app.domain.entities import BhavcopyRecord, StockMasterRecord
from app.repositories.historical_price_repository import SqlAlchemyHistoricalPriceRepository
from app.repositories.market_mover_repository import SqlAlchemyMarketMoverRepository
from app.repositories.stock_repository import SqlAlchemyStockRepository
from tests.conftest import requires_docker

pytestmark = requires_docker


async def _seed_bars(db_session, symbol: str, closes: dict[date, tuple[Decimal, int]]) -> None:
    stock_repo = SqlAlchemyStockRepository(db_session)
    await stock_repo.upsert_universe(
        [
            StockMasterRecord(
                symbol=symbol, isin=None, name=f"{symbol} Ltd", series="EQ", listing_date=None, face_value=None
            )
        ]
    )
    await db_session.commit()

    price_repo = SqlAlchemyHistoricalPriceRepository(db_session)
    records = [
        BhavcopyRecord(symbol=symbol, trade_date=d, open=close, high=close, low=close, close=close, volume=volume)
        for d, (close, volume) in closes.items()
    ]
    await price_repo.bulk_upsert_bars(records)


async def test_get_top_movers_ranks_gainers_and_losers_by_period_change(db_session):
    await _seed_bars(
        db_session,
        "UP",
        {date(2026, 7, 20): (Decimal("100.00"), 500), date(2026, 7, 21): (Decimal("110.00"), 500)},
    )
    await _seed_bars(
        db_session,
        "DOWN",
        {date(2026, 7, 20): (Decimal("100.00"), 500), date(2026, 7, 21): (Decimal("90.00"), 500)},
    )

    repo = SqlAlchemyMarketMoverRepository(db_session)

    gainers = await repo.get_top_movers("gainers", lookback_sessions=1, limit=10)
    losers = await repo.get_top_movers("losers", lookback_sessions=1, limit=10)

    assert gainers[0].symbol == "UP"
    assert abs(gainers[0].change_percent - Decimal("10")) < Decimal("0.01")
    assert losers[0].symbol == "DOWN"
    assert abs(losers[0].change_percent - Decimal("-10")) < Decimal("0.01")


async def test_get_top_movers_excludes_stocks_without_enough_history(db_session):
    await _seed_bars(db_session, "NEWLIST", {date(2026, 7, 21): (Decimal("100.00"), 500)})

    repo = SqlAlchemyMarketMoverRepository(db_session)
    gainers = await repo.get_top_movers("gainers", lookback_sessions=1, limit=10)

    assert "NEWLIST" not in [m.symbol for m in gainers]


async def test_get_most_active_ranks_by_latest_volume(db_session):
    await _seed_bars(
        db_session,
        "QUIET",
        {date(2026, 7, 20): (Decimal("50.00"), 100), date(2026, 7, 21): (Decimal("50.00"), 100)},
    )
    await _seed_bars(
        db_session,
        "BUSY",
        {date(2026, 7, 20): (Decimal("50.00"), 100), date(2026, 7, 21): (Decimal("50.00"), 9_999)},
    )

    repo = SqlAlchemyMarketMoverRepository(db_session)
    result = await repo.get_most_active(limit=10)

    assert result[0].symbol == "BUSY"
    assert result[0].volume == 9_999


async def test_get_52_week_extremes_finds_new_highs_and_lows(db_session):
    await _seed_bars(
        db_session,
        "NEWHIGH",
        {
            date(2026, 7, 19): (Decimal("90.00"), 100),
            date(2026, 7, 20): (Decimal("95.00"), 100),
            date(2026, 7, 21): (Decimal("100.00"), 100),
        },
    )
    await _seed_bars(
        db_session,
        "NEWLOW",
        {
            date(2026, 7, 19): (Decimal("100.00"), 100),
            date(2026, 7, 20): (Decimal("95.00"), 100),
            date(2026, 7, 21): (Decimal("90.00"), 100),
        },
    )
    await _seed_bars(
        db_session,
        "MIDPACK",
        {
            date(2026, 7, 19): (Decimal("100.00"), 100),
            date(2026, 7, 20): (Decimal("110.00"), 100),
            date(2026, 7, 21): (Decimal("105.00"), 100),
        },
    )

    repo = SqlAlchemyMarketMoverRepository(db_session)
    highs = await repo.get_52_week_extremes("high", limit=10)
    lows = await repo.get_52_week_extremes("low", limit=10)

    assert "NEWHIGH" in [m.symbol for m in highs]
    assert "MIDPACK" not in [m.symbol for m in highs]
    assert "NEWLOW" in [m.symbol for m in lows]
    assert "MIDPACK" not in [m.symbol for m in lows]
