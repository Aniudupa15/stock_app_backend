from datetime import date, timedelta
from decimal import Decimal

import pytest

from app.domain.entities import FinancialResultRecord, InstrumentType, OhlcvBar, Stock
from app.services.fundamentals_service import FundamentalsService
from app.services.intraday_signal_service import IntradaySignalService
from app.services.long_term_signal_service import LongTermSignalService
from app.services.signal_snapshot_sync_service import SignalSnapshotSyncService, _investment_tenure
from tests.conftest import (
    FakeCorporateActionRepository,
    FakeFinancialResultRepository,
    FakeHistoricalPriceRepository,
    FakeIntradaySignalSnapshotRepository,
    FakeLongTermSignalSnapshotRepository,
    FakeStockRepository,
)


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


def _bars(closes: list[float]) -> list[OhlcvBar]:
    today = date.today()
    n = len(closes)
    return [
        OhlcvBar(
            trade_date=today - timedelta(days=(n - 1 - i)),
            open=Decimal(str(c)),
            high=Decimal(str(c + 1)),
            low=Decimal(str(c - 1)),
            close=Decimal(str(c)),
            volume=1000 + i,
        )
        for i, c in enumerate(closes)
    ]


def _quarter(period_end: date, revenue: str, profit: str, eps: str) -> FinancialResultRecord:
    return FinancialResultRecord(
        symbol="RELIANCE",
        period_start=date(period_end.year, 1, 1),
        period_end=period_end,
        consolidated=False,
        revenue=Decimal(revenue),
        profit=Decimal(profit),
        eps_basic=Decimal(eps),
        eps_diluted=Decimal(eps),
    )


@pytest.fixture
def signal_snapshot_service():
    stock_repo = FakeStockRepository([_stock("RELIANCE")])
    price_repo = FakeHistoricalPriceRepository({"RELIANCE": _bars([100 + i for i in range(60)])})
    financial_repo = FakeFinancialResultRepository(
        {
            "RELIANCE": [
                _quarter(date(2026, 3, 31), "1200", "150", "10"),
                _quarter(date(2025, 12, 31), "1100", "130", "9"),
                _quarter(date(2025, 9, 30), "1000", "120", "8"),
                _quarter(date(2025, 6, 30), "900", "100", "7"),
            ]
        }
    )
    corp_action_repo = FakeCorporateActionRepository()
    intraday_snapshot_repo = FakeIntradaySignalSnapshotRepository()
    long_term_snapshot_repo = FakeLongTermSignalSnapshotRepository()

    intraday_signal_service = IntradaySignalService(stock_repo, price_repo)
    fundamentals_service = FundamentalsService(stock_repo, financial_repo, price_repo, corp_action_repo)
    long_term_signal_service = LongTermSignalService(stock_repo, fundamentals_service)

    return (
        SignalSnapshotSyncService(
            stock_repo,
            intraday_signal_service,
            long_term_signal_service,
            intraday_snapshot_repo,
            long_term_snapshot_repo,
        ),
        intraday_snapshot_repo,
        long_term_snapshot_repo,
    )


async def test_sync_intraday_stores_a_snapshot_for_stocks_with_enough_history(signal_snapshot_service):
    service, intraday_repo, _ = signal_snapshot_service

    upserted = await service.sync_intraday()

    assert upserted == 1
    snapshot = intraday_repo.snapshots_by_symbol["RELIANCE"]
    assert snapshot.signal in ("BUY", "SELL", "HOLD")
    assert snapshot.confidence is not None
    assert isinstance(snapshot.reasoning, list)


async def test_sync_intraday_skips_stocks_with_no_price_history():
    stock_repo = FakeStockRepository([_stock("NODATA")])
    price_repo = FakeHistoricalPriceRepository({})
    intraday_snapshot_repo = FakeIntradaySignalSnapshotRepository()
    service = SignalSnapshotSyncService(
        stock_repo,
        IntradaySignalService(stock_repo, price_repo),
        LongTermSignalService(
            stock_repo,
            FundamentalsService(
                stock_repo, FakeFinancialResultRepository(), price_repo, FakeCorporateActionRepository()
            ),
        ),
        intraday_snapshot_repo,
        FakeLongTermSignalSnapshotRepository(),
    )

    upserted = await service.sync_intraday()

    assert upserted == 0
    assert intraday_snapshot_repo.snapshots_by_symbol == {}


async def test_sync_long_term_stores_a_snapshot_with_investment_tenure(signal_snapshot_service):
    service, _, long_term_repo = signal_snapshot_service

    upserted = await service.sync_long_term()

    assert upserted == 1
    snapshot = long_term_repo.snapshots_by_symbol["RELIANCE"]
    assert snapshot.signal in ("BUY", "HOLD", "AVOID")
    assert snapshot.investment_tenure in ("6 Months", "1 Year", "3 Years", "5 Years")


@pytest.mark.parametrize(
    "risk_level,growth_potential,expected",
    [
        ("Low", "High", "1 Year"),
        ("Moderate", "High", "1 Year"),
        ("High", "High", "6 Months"),
        ("Low", "Moderate", "3 Years"),
        ("Moderate", "Low/Uncertain", "5 Years"),
        ("High", "Low/Uncertain", "5 Years"),
    ],
)
def test_investment_tenure_derivation(risk_level, growth_potential, expected):
    assert _investment_tenure(risk_level, growth_potential) == expected
