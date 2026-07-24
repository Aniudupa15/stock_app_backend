from datetime import date
from decimal import Decimal

from app.domain.entities import IntradaySignalSnapshot, LongTermSignalSnapshot
from app.services.analysis_service import AnalysisService
from tests.conftest import FakeIntradaySignalSnapshotRepository, FakeLongTermSignalSnapshotRepository


def _intraday(symbol: str, signal: str, confidence: str) -> IntradaySignalSnapshot:
    return IntradaySignalSnapshot(
        symbol=symbol,
        name=f"{symbol} Ltd",
        as_of=date.today(),
        signal=signal,
        confidence=Decimal(confidence),
        entry_price=Decimal("100"),
        target_price=Decimal("110"),
        stop_loss=Decimal("95"),
        reasoning=["RSI is oversold at 25.00"],
    )


def _long_term(symbol: str, signal: str, confidence: int, tenure: str) -> LongTermSignalSnapshot:
    return LongTermSignalSnapshot(
        symbol=symbol,
        name=f"{symbol} Ltd",
        as_of=date.today(),
        signal=signal,
        confidence=confidence,
        risk_level="Low",
        growth_potential="High",
        investment_tenure=tenure,
        reasoning=["Revenue grew 15% YoY"],
    )


async def test_get_top_intraday_excludes_hold_and_filters_by_confidence():
    repo = FakeIntradaySignalSnapshotRepository(
        [
            _intraday("AAA", "BUY", "80"),
            _intraday("BBB", "HOLD", "90"),
            _intraday("CCC", "SELL", "30"),
        ]
    )
    service = AnalysisService(repo, FakeLongTermSignalSnapshotRepository())

    results = await service.get_top_intraday(Decimal("50"), 50)

    assert [r.symbol for r in results] == ["AAA"]


async def test_get_top_long_term_filters_by_tenure():
    repo = FakeLongTermSignalSnapshotRepository(
        [
            _long_term("AAA", "BUY", 80, "1 Year"),
            _long_term("BBB", "BUY", 90, "5 Years"),
            _long_term("CCC", "HOLD", 95, "1 Year"),
        ]
    )
    service = AnalysisService(FakeIntradaySignalSnapshotRepository(), repo)

    results = await service.get_top_long_term(0, "1 Year", 50)

    assert [r.symbol for r in results] == ["AAA"]


async def test_get_top_long_term_orders_by_confidence_descending():
    repo = FakeLongTermSignalSnapshotRepository(
        [
            _long_term("AAA", "BUY", 60, "1 Year"),
            _long_term("BBB", "BUY", 90, "1 Year"),
        ]
    )
    service = AnalysisService(FakeIntradaySignalSnapshotRepository(), repo)

    results = await service.get_top_long_term(0, None, 50)

    assert [r.symbol for r in results] == ["BBB", "AAA"]
