import logging
from datetime import date

from app.domain.entities import IntradaySignalSnapshot, LongTermSignalSnapshot
from app.domain.ports import (
    IntradaySignalSnapshotRepositoryPort,
    LongTermSignalSnapshotRepositoryPort,
    StockRepositoryPort,
)
from app.schemas.intraday_signal import IntradaySignalOut
from app.schemas.long_term_signal import LongTermSignalOut
from app.services.intraday_signal_service import IntradaySignalService
from app.services.long_term_signal_service import LongTermSignalService

logger = logging.getLogger(__name__)

_STRONG_GROWTH = "High"
_LOW_RISK = "Low"
_HIGH_RISK = "High"


def _investment_tenure(risk_level: str, growth_potential: str) -> str:
    """Deterministic rule, same "no ML" philosophy as every other signal in
    this app: strong growth at low/moderate risk can be realized sooner;
    strong growth at high risk needs a shorter, more tactical horizon to
    de-risk; anything uncertain or weak gets the longest horizon to let the
    thesis play out (or not).
    """
    if growth_potential == _STRONG_GROWTH:
        return "6 Months" if risk_level == _HIGH_RISK else "1 Year"
    if risk_level == _LOW_RISK:
        return "3 Years"
    return "5 Years"


class SignalSnapshotSyncService:
    """Refreshes the materialized `intraday_signal_snapshots`/
    `long_term_signal_snapshots` tables the Analysis screen reads from -
    same rationale and shape as `IndicatorSnapshotSyncService` for the
    screener. Reuses `IntradaySignalService`/`LongTermSignalService`'s
    existing, unmodified per-symbol logic across every active stock - not a
    duplicated scoring path. Both underlying signal services are pure DB
    reads + local computation (no live NSE calls), confirmed safe to run in
    a loop across the whole universe.
    """

    def __init__(
        self,
        stock_repository: StockRepositoryPort,
        intraday_signal_service: IntradaySignalService,
        long_term_signal_service: LongTermSignalService,
        intraday_snapshot_repository: IntradaySignalSnapshotRepositoryPort,
        long_term_snapshot_repository: LongTermSignalSnapshotRepositoryPort,
    ):
        self._stock_repository = stock_repository
        self._intraday_signal_service = intraday_signal_service
        self._long_term_signal_service = long_term_signal_service
        self._intraday_snapshot_repository = intraday_snapshot_repository
        self._long_term_snapshot_repository = long_term_snapshot_repository

    async def sync_intraday(self) -> int:
        today = date.today()
        symbols = await self._stock_repository.list_active_symbols()
        snapshots: list[IntradaySignalSnapshot] = []
        for symbol in symbols:
            try:
                out = await self._intraday_signal_service.get_signal(symbol)
            except Exception:
                logger.warning("Intraday signal snapshot: skipping %s (unexpected error)", symbol, exc_info=True)
                continue
            if not out.has_data:
                continue
            snapshots.append(self._to_intraday_snapshot(symbol, out, today))
        return await self._intraday_snapshot_repository.bulk_upsert(snapshots)

    async def sync_long_term(self) -> int:
        today = date.today()
        symbols = await self._stock_repository.list_active_symbols()
        snapshots: list[LongTermSignalSnapshot] = []
        for symbol in symbols:
            try:
                out = await self._long_term_signal_service.get_signal(symbol)
            except Exception:
                logger.warning("Long-term signal snapshot: skipping %s (unexpected error)", symbol, exc_info=True)
                continue
            if not out.has_data:
                continue
            snapshots.append(self._to_long_term_snapshot(symbol, out, today))
        return await self._long_term_snapshot_repository.bulk_upsert(snapshots)

    def _to_intraday_snapshot(self, symbol: str, out: IntradaySignalOut, as_of: date) -> IntradaySignalSnapshot:
        return IntradaySignalSnapshot(
            symbol=symbol,
            name=symbol,
            as_of=as_of,
            signal=out.signal,
            confidence=out.confidence,
            entry_price=out.entry_price,
            target_price=out.target_price,
            stop_loss=out.stop_loss,
            reasoning=out.reasoning,
        )

    def _to_long_term_snapshot(self, symbol: str, out: LongTermSignalOut, as_of: date) -> LongTermSignalSnapshot:
        return LongTermSignalSnapshot(
            symbol=symbol,
            name=symbol,
            as_of=as_of,
            signal=out.signal,
            confidence=out.confidence,
            risk_level=out.risk_level,
            growth_potential=out.growth_potential,
            investment_tenure=_investment_tenure(out.risk_level, out.growth_potential),
            reasoning=out.reasoning,
        )
