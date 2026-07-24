from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import LongTermSignalSnapshot
from app.domain.ports import LongTermSignalSnapshotRepositoryPort
from app.models.long_term_signal_snapshot import LongTermSignalSnapshotModel
from app.models.stock import StockModel

_UPSERT_BATCH_SIZE = 500


class SqlAlchemyLongTermSignalSnapshotRepository(LongTermSignalSnapshotRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def bulk_upsert(self, snapshots: list[LongTermSignalSnapshot]) -> int:
        if not snapshots:
            return 0

        symbols = {s.symbol for s in snapshots}
        stmt = select(StockModel.id, StockModel.symbol).where(StockModel.symbol.in_(symbols))
        result = await self._session.execute(stmt)
        symbol_to_id = {row.symbol: row.id for row in result}

        rows = [
            {
                "stock_id": symbol_to_id[s.symbol],
                "as_of": s.as_of,
                "signal": s.signal,
                "confidence": s.confidence,
                "risk_level": s.risk_level,
                "growth_potential": s.growth_potential,
                "investment_tenure": s.investment_tenure,
                "reasoning": s.reasoning,
            }
            for s in snapshots
            if s.symbol in symbol_to_id
        ]

        upserted = 0
        for i in range(0, len(rows), _UPSERT_BATCH_SIZE):
            batch = rows[i : i + _UPSERT_BATCH_SIZE]
            stmt = pg_insert(LongTermSignalSnapshotModel).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=["stock_id"],
                set_={
                    "as_of": stmt.excluded.as_of,
                    "signal": stmt.excluded.signal,
                    "confidence": stmt.excluded.confidence,
                    "risk_level": stmt.excluded.risk_level,
                    "growth_potential": stmt.excluded.growth_potential,
                    "investment_tenure": stmt.excluded.investment_tenure,
                    "reasoning": stmt.excluded.reasoning,
                },
            )
            await self._session.execute(stmt)
            upserted += len(batch)

        await self._session.commit()
        return upserted

    async def list_top(self, min_confidence: int, tenure: str | None, limit: int) -> list[LongTermSignalSnapshot]:
        stmt = (
            select(
                StockModel.symbol,
                StockModel.name,
                LongTermSignalSnapshotModel.as_of,
                LongTermSignalSnapshotModel.signal,
                LongTermSignalSnapshotModel.confidence,
                LongTermSignalSnapshotModel.risk_level,
                LongTermSignalSnapshotModel.growth_potential,
                LongTermSignalSnapshotModel.investment_tenure,
                LongTermSignalSnapshotModel.reasoning,
            )
            .join(StockModel, StockModel.id == LongTermSignalSnapshotModel.stock_id)
            .where(
                StockModel.is_active.is_(True),
                LongTermSignalSnapshotModel.signal == "BUY",
                LongTermSignalSnapshotModel.confidence >= min_confidence,
            )
        )
        if tenure is not None:
            stmt = stmt.where(LongTermSignalSnapshotModel.investment_tenure == tenure)

        stmt = stmt.order_by(LongTermSignalSnapshotModel.confidence.desc()).limit(limit)

        result = await self._session.execute(stmt)
        return [
            LongTermSignalSnapshot(
                symbol=row.symbol,
                name=row.name,
                as_of=row.as_of,
                signal=row.signal,
                confidence=row.confidence,
                risk_level=row.risk_level,
                growth_potential=row.growth_potential,
                investment_tenure=row.investment_tenure,
                reasoning=row.reasoning,
            )
            for row in result
        ]
