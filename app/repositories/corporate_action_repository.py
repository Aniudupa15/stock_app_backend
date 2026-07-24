from datetime import date

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import CorporateAction
from app.domain.ports import CorporateActionRepositoryPort
from app.models.corporate_action import CorporateActionModel
from app.models.stock import StockModel

_UPSERT_BATCH_SIZE = 500


class SqlAlchemyCorporateActionRepository(CorporateActionRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def bulk_upsert(self, records: list[CorporateAction]) -> int:
        if not records:
            return 0

        symbols = {r.symbol for r in records}
        stmt = select(StockModel.id, StockModel.symbol).where(StockModel.symbol.in_(symbols))
        result = await self._session.execute(stmt)
        symbol_to_id = {row.symbol: row.id for row in result}

        rows = [
            {
                "stock_id": symbol_to_id[r.symbol],
                "purpose": r.purpose,
                "face_value": r.face_value,
                "ex_date": r.ex_date,
                "record_date": r.record_date,
                "book_closure_start": r.book_closure_start,
                "book_closure_end": r.book_closure_end,
            }
            for r in records
            if r.symbol in symbol_to_id
        ]

        upserted = 0
        for i in range(0, len(rows), _UPSERT_BATCH_SIZE):
            batch = rows[i : i + _UPSERT_BATCH_SIZE]
            stmt = pg_insert(CorporateActionModel).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=["stock_id", "purpose", "ex_date"],
                set_={
                    "face_value": stmt.excluded.face_value,
                    "record_date": stmt.excluded.record_date,
                    "book_closure_start": stmt.excluded.book_closure_start,
                    "book_closure_end": stmt.excluded.book_closure_end,
                },
            )
            await self._session.execute(stmt)
            upserted += len(batch)

        await self._session.commit()
        return upserted

    async def get_for_symbol(self, symbol: str) -> list[CorporateAction]:
        stmt = (
            select(CorporateActionModel)
            .join(StockModel, StockModel.id == CorporateActionModel.stock_id)
            .where(StockModel.symbol == symbol.strip().upper())
            .order_by(CorporateActionModel.ex_date.desc().nulls_last())
        )
        result = await self._session.execute(stmt)
        return [
            CorporateAction(
                symbol=symbol.strip().upper(),
                purpose=row.purpose,
                face_value=row.face_value,
                ex_date=row.ex_date,
                record_date=row.record_date,
                book_closure_start=row.book_closure_start,
                book_closure_end=row.book_closure_end,
            )
            for row in result.scalars().all()
        ]

    async def list_dividend_actions(self, ex_date_from: date, ex_date_to: date) -> list[CorporateAction]:
        stmt = (
            select(CorporateActionModel, StockModel.symbol)
            .join(StockModel, StockModel.id == CorporateActionModel.stock_id)
            .where(
                CorporateActionModel.purpose.ilike("%dividend%"),
                CorporateActionModel.ex_date.is_not(None),
                CorporateActionModel.ex_date >= ex_date_from,
                CorporateActionModel.ex_date <= ex_date_to,
                StockModel.is_active.is_(True),
            )
            .order_by(CorporateActionModel.ex_date.asc())
        )
        result = await self._session.execute(stmt)
        return [
            CorporateAction(
                symbol=symbol,
                purpose=action.purpose,
                face_value=action.face_value,
                ex_date=action.ex_date,
                record_date=action.record_date,
                book_closure_start=action.book_closure_start,
                book_closure_end=action.book_closure_end,
            )
            for action, symbol in result.all()
        ]
