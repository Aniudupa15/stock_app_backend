from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import IntradaySignalSnapshot
from app.domain.ports import IntradaySignalSnapshotRepositoryPort
from app.models.intraday_signal_snapshot import IntradaySignalSnapshotModel
from app.models.stock import StockModel

_UPSERT_BATCH_SIZE = 500


class SqlAlchemyIntradaySignalSnapshotRepository(IntradaySignalSnapshotRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def bulk_upsert(self, snapshots: list[IntradaySignalSnapshot]) -> int:
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
                "entry_price": s.entry_price,
                "target_price": s.target_price,
                "stop_loss": s.stop_loss,
                "reasoning": s.reasoning,
            }
            for s in snapshots
            if s.symbol in symbol_to_id
        ]

        upserted = 0
        for i in range(0, len(rows), _UPSERT_BATCH_SIZE):
            batch = rows[i : i + _UPSERT_BATCH_SIZE]
            stmt = pg_insert(IntradaySignalSnapshotModel).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=["stock_id"],
                set_={
                    "as_of": stmt.excluded.as_of,
                    "signal": stmt.excluded.signal,
                    "confidence": stmt.excluded.confidence,
                    "entry_price": stmt.excluded.entry_price,
                    "target_price": stmt.excluded.target_price,
                    "stop_loss": stmt.excluded.stop_loss,
                    "reasoning": stmt.excluded.reasoning,
                },
            )
            await self._session.execute(stmt)
            upserted += len(batch)

        await self._session.commit()
        return upserted

    async def list_top(self, min_confidence: Decimal, limit: int) -> list[IntradaySignalSnapshot]:
        stmt = (
            select(
                StockModel.symbol,
                StockModel.name,
                IntradaySignalSnapshotModel.as_of,
                IntradaySignalSnapshotModel.signal,
                IntradaySignalSnapshotModel.confidence,
                IntradaySignalSnapshotModel.entry_price,
                IntradaySignalSnapshotModel.target_price,
                IntradaySignalSnapshotModel.stop_loss,
                IntradaySignalSnapshotModel.reasoning,
            )
            .join(StockModel, StockModel.id == IntradaySignalSnapshotModel.stock_id)
            .where(
                StockModel.is_active.is_(True),
                IntradaySignalSnapshotModel.signal.in_(["BUY", "SELL"]),
                IntradaySignalSnapshotModel.confidence >= min_confidence,
            )
            .order_by(IntradaySignalSnapshotModel.confidence.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [
            IntradaySignalSnapshot(
                symbol=row.symbol,
                name=row.name,
                as_of=row.as_of,
                signal=row.signal,
                confidence=row.confidence,
                entry_price=row.entry_price,
                target_price=row.target_price,
                stop_loss=row.stop_loss,
                reasoning=row.reasoning,
            )
            for row in result
        ]
