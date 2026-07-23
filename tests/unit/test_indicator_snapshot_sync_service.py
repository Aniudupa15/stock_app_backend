from datetime import date, timedelta
from decimal import Decimal

from app.domain.entities import InstrumentType, OhlcvBar, Stock
from app.services.indicator_snapshot_sync_service import IndicatorSnapshotSyncService
from tests.conftest import FakeHistoricalPriceRepository, FakeScreenerRepository, FakeStockRepository


def _stock(symbol: str) -> Stock:
    return Stock(
        symbol=symbol,
        isin=None,
        name=f"{symbol} Ltd",
        series="EQ",
        sector=None,
        industry=None,
        instrument_type=InstrumentType.EQUITY,
        listing_date=None,
        face_value=None,
        is_active=True,
    )


def _bars(symbol: str, closes: list[float]) -> list[OhlcvBar]:
    today = date.today()
    n = len(closes)
    return [
        OhlcvBar(
            trade_date=today - timedelta(days=(n - 1 - i)),
            open=Decimal(str(c)),
            high=Decimal(str(c)),
            low=Decimal(str(c)),
            close=Decimal(str(c)),
            volume=1000,
        )
        for i, c in enumerate(closes)
    ]


async def test_sync_all_computes_and_upserts_snapshots():
    stock_repo = FakeStockRepository([_stock("RELIANCE")])
    price_repo = FakeHistoricalPriceRepository({"RELIANCE": _bars("RELIANCE", [100 + i for i in range(60)])})
    screener_repo = FakeScreenerRepository()
    service = IndicatorSnapshotSyncService(stock_repo, price_repo, screener_repo)

    upserted = await service.sync_all()

    assert upserted == 1
    snapshot = screener_repo.snapshots_by_symbol["RELIANCE"]
    assert snapshot.close == Decimal("159")
    assert snapshot.rsi_14 is not None
    assert snapshot.sma_50 is not None
    assert snapshot.sma_200 is None  # not enough history (only 60 bars)


async def test_sync_all_skips_stocks_with_no_price_history():
    stock_repo = FakeStockRepository([_stock("NODATA")])
    price_repo = FakeHistoricalPriceRepository({})
    screener_repo = FakeScreenerRepository()
    service = IndicatorSnapshotSyncService(stock_repo, price_repo, screener_repo)

    upserted = await service.sync_all()

    assert upserted == 0
    assert screener_repo.snapshots_by_symbol == {}
