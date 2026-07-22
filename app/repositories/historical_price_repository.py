from datetime import date

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import BhavcopyRecord, OhlcvBar
from app.domain.ports import HistoricalPriceRepositoryPort
from app.models.historical_price import HistoricalPriceModel
from app.models.stock import StockModel

_UPSERT_BATCH_SIZE = 500


class SqlAlchemyHistoricalPriceRepository(HistoricalPriceRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def bulk_upsert_bars(self, records: list[BhavcopyRecord]) -> int:
        if not records:
            return 0

        symbols = {r.symbol for r in records}
        stmt = select(StockModel.id, StockModel.symbol).where(StockModel.symbol.in_(symbols))
        result = await self._session.execute(stmt)
        symbol_to_id = {row.symbol: row.id for row in result}

        # Records for symbols not in `stocks` (SGBs, T-Bills, delisted names,
        # etc. mixed into the full Cash Market Bhavcopy) are silently skipped -
        # this is the authoritative equity filter, not a guess at NSE's series codes.
        rows = [
            {
                "stock_id": symbol_to_id[r.symbol],
                "trade_date": r.trade_date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in records
            if r.symbol in symbol_to_id
        ]

        upserted = 0
        for i in range(0, len(rows), _UPSERT_BATCH_SIZE):
            batch = rows[i : i + _UPSERT_BATCH_SIZE]
            stmt = pg_insert(HistoricalPriceModel).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=["stock_id", "trade_date"],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                },
            )
            await self._session.execute(stmt)
            upserted += len(batch)

        await self._session.commit()
        return upserted

    async def get_bars(self, symbol: str, from_date: date, to_date: date) -> list[OhlcvBar]:
        stmt = (
            select(HistoricalPriceModel)
            .join(StockModel, StockModel.id == HistoricalPriceModel.stock_id)
            .where(
                StockModel.symbol == symbol.strip().upper(),
                HistoricalPriceModel.trade_date >= from_date,
                HistoricalPriceModel.trade_date <= to_date,
            )
            .order_by(HistoricalPriceModel.trade_date.asc())
        )
        result = await self._session.execute(stmt)
        return [
            OhlcvBar(
                trade_date=row.trade_date,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
            )
            for row in result.scalars().all()
        ]
