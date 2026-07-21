from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import InstrumentType, Stock, StockMasterRecord
from app.domain.ports import StockRepositoryPort
from app.models.stock import StockModel

_UPSERT_BATCH_SIZE = 500


def _to_entity(row: StockModel) -> Stock:
    return Stock(
        symbol=row.symbol,
        isin=row.isin,
        name=row.name,
        series=row.series,
        sector=row.sector,
        industry=row.industry,
        instrument_type=row.instrument_type,
        listing_date=row.listing_date,
        face_value=row.face_value,
        is_active=row.is_active,
    )


class SqlAlchemyStockRepository(StockRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_symbol(self, symbol: str) -> Stock | None:
        stmt = select(StockModel).where(
            StockModel.symbol == symbol.strip().upper(), StockModel.is_active.is_(True)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return _to_entity(row) if row else None

    async def search_by_symbol_or_name(self, query: str, limit: int) -> list[Stock]:
        normalized = query.strip()
        symbol_pattern = f"{normalized.upper()}%"
        name_pattern = f"%{normalized.lower()}%"
        stmt = (
            select(StockModel)
            .where(
                StockModel.is_active.is_(True),
                or_(StockModel.symbol.ilike(symbol_pattern), func.lower(StockModel.name).like(name_pattern)),
            )
            .order_by((StockModel.symbol == normalized.upper()).desc(), StockModel.symbol)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        stocks = [_to_entity(row) for row in result.scalars().all()]

        if len(stocks) >= limit:
            return stocks

        # Fuzzy fallback (pg_trgm) for typos/partial matches the exact-prefix
        # and substring query above missed - only runs when that fast path
        # came up short, so well-formed queries never pay for it. Checks both
        # symbol and name similarity (a typo could land in either, e.g.
        # "RELAINCE" is a symbol typo, "Infosis" is a name typo).
        seen_symbols = {s.symbol for s in stocks}
        similarity = func.greatest(
            func.similarity(StockModel.symbol, normalized.upper()), func.similarity(StockModel.name, normalized)
        )
        fuzzy_stmt = (
            select(StockModel)
            .where(
                StockModel.is_active.is_(True),
                similarity > 0.2,
                StockModel.symbol.notin_(seen_symbols),
            )
            .order_by(similarity.desc())
            .limit(limit - len(stocks))
        )
        fuzzy_result = await self._session.execute(fuzzy_stmt)
        stocks.extend(_to_entity(row) for row in fuzzy_result.scalars().all())
        return stocks

    async def upsert_universe(self, records: list[StockMasterRecord]) -> int:
        affected = 0
        for i in range(0, len(records), _UPSERT_BATCH_SIZE):
            batch = records[i : i + _UPSERT_BATCH_SIZE]
            values = [
                {
                    "symbol": r.symbol,
                    "isin": r.isin,
                    "name": r.name,
                    "series": r.series,
                    "instrument_type": InstrumentType.EQUITY,
                    "listing_date": r.listing_date,
                    "face_value": r.face_value,
                    "is_active": True,
                }
                for r in batch
            ]
            stmt = pg_insert(StockModel).values(values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol"],
                set_={
                    "isin": stmt.excluded.isin,
                    "name": stmt.excluded.name,
                    "series": stmt.excluded.series,
                    "listing_date": stmt.excluded.listing_date,
                    "face_value": stmt.excluded.face_value,
                    "is_active": True,
                    "updated_at": func.now(),
                },
            )
            await self._session.execute(stmt)
            affected += len(batch)
        await self._session.commit()
        return affected

    async def deactivate_missing(self, active_symbols: set[str]) -> int:
        stmt = (
            update(StockModel)
            .where(StockModel.is_active.is_(True), StockModel.symbol.notin_(active_symbols))
            .values(is_active=False)
        )
        result = await self._session.execute(stmt)
        await self._session.commit()
        return result.rowcount or 0
