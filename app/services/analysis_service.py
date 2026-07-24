from decimal import Decimal

from app.domain.ports import IntradaySignalSnapshotRepositoryPort, LongTermSignalSnapshotRepositoryPort
from app.schemas.analysis import IntradayRecommendationOut, LongTermRecommendationOut


class AnalysisService:
    """Read-only - the Analysis screen's "top picks" lists are served
    entirely from the materialized snapshot tables `SignalSnapshotSyncService`
    refreshes daily. Same read/write split as `ScreenerService` vs
    `IndicatorSnapshotSyncService`.
    """

    def __init__(
        self,
        intraday_snapshot_repository: IntradaySignalSnapshotRepositoryPort,
        long_term_snapshot_repository: LongTermSignalSnapshotRepositoryPort,
    ):
        self._intraday_snapshot_repository = intraday_snapshot_repository
        self._long_term_snapshot_repository = long_term_snapshot_repository

    async def get_top_intraday(self, min_confidence: Decimal, limit: int) -> list[IntradayRecommendationOut]:
        snapshots = await self._intraday_snapshot_repository.list_top(min_confidence, limit)
        return [
            IntradayRecommendationOut(
                symbol=s.symbol,
                name=s.name,
                as_of=s.as_of,
                signal=s.signal,
                confidence=s.confidence,
                entry_price=s.entry_price,
                target_price=s.target_price,
                stop_loss=s.stop_loss,
                reasoning=s.reasoning,
            )
            for s in snapshots
        ]

    async def get_top_long_term(
        self, min_confidence: int, tenure: str | None, limit: int
    ) -> list[LongTermRecommendationOut]:
        snapshots = await self._long_term_snapshot_repository.list_top(min_confidence, tenure, limit)
        return [
            LongTermRecommendationOut(
                symbol=s.symbol,
                name=s.name,
                as_of=s.as_of,
                signal=s.signal,
                confidence=s.confidence,
                risk_level=s.risk_level,
                growth_potential=s.growth_potential,
                investment_tenure=s.investment_tenure,
                reasoning=s.reasoning,
            )
            for s in snapshots
        ]
