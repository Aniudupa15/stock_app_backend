from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import FinancialResultRecord
from app.domain.ports import FinancialResultRepositoryPort
from app.models.financial_result import FinancialResultModel
from app.models.stock import StockModel

_UPSERT_BATCH_SIZE = 500


class SqlAlchemyFinancialResultRepository(FinancialResultRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def bulk_upsert(self, records: list[FinancialResultRecord]) -> int:
        if not records:
            return 0

        symbols = {r.symbol for r in records}
        stmt = select(StockModel.id, StockModel.symbol).where(StockModel.symbol.in_(symbols))
        result = await self._session.execute(stmt)
        symbol_to_id = {row.symbol: row.id for row in result}

        rows = [
            {
                "stock_id": symbol_to_id[r.symbol],
                "period_start": r.period_start,
                "period_end": r.period_end,
                "consolidated": r.consolidated,
                "revenue": r.revenue,
                "profit": r.profit,
                "eps_basic": r.eps_basic,
                "eps_diluted": r.eps_diluted,
            }
            for r in records
            if r.symbol in symbol_to_id
        ]

        upserted = 0
        for i in range(0, len(rows), _UPSERT_BATCH_SIZE):
            batch = rows[i : i + _UPSERT_BATCH_SIZE]
            stmt = pg_insert(FinancialResultModel).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=["stock_id", "period_end", "consolidated"],
                set_={
                    "period_start": stmt.excluded.period_start,
                    "revenue": stmt.excluded.revenue,
                    "profit": stmt.excluded.profit,
                    "eps_basic": stmt.excluded.eps_basic,
                    "eps_diluted": stmt.excluded.eps_diluted,
                },
            )
            await self._session.execute(stmt)
            upserted += len(batch)

        await self._session.commit()
        return upserted

    async def get_recent_quarters(self, symbol: str, consolidated: bool, limit: int) -> list[FinancialResultRecord]:
        stmt = (
            select(FinancialResultModel)
            .join(StockModel, StockModel.id == FinancialResultModel.stock_id)
            .where(StockModel.symbol == symbol.strip().upper(), FinancialResultModel.consolidated == consolidated)
            .order_by(FinancialResultModel.period_end.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [
            FinancialResultRecord(
                symbol=symbol.strip().upper(),
                period_start=row.period_start,
                period_end=row.period_end,
                consolidated=row.consolidated,
                revenue=row.revenue,
                profit=row.profit,
                eps_basic=row.eps_basic,
                eps_diluted=row.eps_diluted,
            )
            for row in result.scalars().all()
        ]
