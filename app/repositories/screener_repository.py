from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import ScreenerFilters, StockIndicatorSnapshot
from app.domain.ports import ScreenerRepositoryPort
from app.models.stock import StockModel
from app.models.stock_indicator_snapshot import StockIndicatorSnapshotModel

_UPSERT_BATCH_SIZE = 500


class SqlAlchemyScreenerRepository(ScreenerRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def bulk_upsert(self, snapshots: list[StockIndicatorSnapshot]) -> int:
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
                "close": s.close,
                "volume": s.volume,
                "rsi_14": s.rsi_14,
                "sma_50": s.sma_50,
                "sma_200": s.sma_200,
            }
            for s in snapshots
            if s.symbol in symbol_to_id
        ]

        upserted = 0
        for i in range(0, len(rows), _UPSERT_BATCH_SIZE):
            batch = rows[i : i + _UPSERT_BATCH_SIZE]
            stmt = pg_insert(StockIndicatorSnapshotModel).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=["stock_id"],
                set_={
                    "as_of": stmt.excluded.as_of,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                    "rsi_14": stmt.excluded.rsi_14,
                    "sma_50": stmt.excluded.sma_50,
                    "sma_200": stmt.excluded.sma_200,
                },
            )
            await self._session.execute(stmt)
            upserted += len(batch)

        await self._session.commit()
        return upserted

    async def screen(self, filters: ScreenerFilters, limit: int) -> list[StockIndicatorSnapshot]:
        stmt = (
            select(
                StockModel.symbol,
                StockModel.name,
                StockIndicatorSnapshotModel.as_of,
                StockIndicatorSnapshotModel.close,
                StockIndicatorSnapshotModel.volume,
                StockIndicatorSnapshotModel.rsi_14,
                StockIndicatorSnapshotModel.sma_50,
                StockIndicatorSnapshotModel.sma_200,
            )
            .join(StockModel, StockModel.id == StockIndicatorSnapshotModel.stock_id)
            .where(StockModel.is_active.is_(True))
        )

        if filters.rsi_below is not None:
            stmt = stmt.where(StockIndicatorSnapshotModel.rsi_14 < filters.rsi_below)
        if filters.rsi_above is not None:
            stmt = stmt.where(StockIndicatorSnapshotModel.rsi_14 > filters.rsi_above)
        if filters.price_min is not None:
            stmt = stmt.where(StockIndicatorSnapshotModel.close >= filters.price_min)
        if filters.price_max is not None:
            stmt = stmt.where(StockIndicatorSnapshotModel.close <= filters.price_max)
        if filters.above_sma_50 is True:
            stmt = stmt.where(StockIndicatorSnapshotModel.close > StockIndicatorSnapshotModel.sma_50)
        elif filters.above_sma_50 is False:
            stmt = stmt.where(StockIndicatorSnapshotModel.close < StockIndicatorSnapshotModel.sma_50)
        if filters.min_volume is not None:
            stmt = stmt.where(StockIndicatorSnapshotModel.volume >= filters.min_volume)

        stmt = stmt.order_by(StockModel.symbol.asc()).limit(limit)

        result = await self._session.execute(stmt)
        return [
            StockIndicatorSnapshot(
                symbol=row.symbol,
                name=row.name,
                as_of=row.as_of,
                close=row.close,
                volume=row.volume,
                rsi_14=row.rsi_14,
                sma_50=row.sma_50,
                sma_200=row.sma_200,
            )
            for row in result
        ]
