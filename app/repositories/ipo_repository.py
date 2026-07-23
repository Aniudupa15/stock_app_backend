from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import IpoFiling
from app.domain.ports import IpoRepositoryPort
from app.models.ipo_filing import IpoFilingModel

_UPSERT_BATCH_SIZE = 200


def _to_entity(row: IpoFilingModel) -> IpoFiling:
    return IpoFiling(
        symbol=row.symbol,
        company_name=row.company_name,
        status=row.status,
        price_range=row.price_range,
        issue_size=row.issue_size,
        issue_start_date=row.issue_start_date,
        issue_end_date=row.issue_end_date,
        listing_date=row.listing_date,
        series=row.series,
    )


class SqlAlchemyIpoRepository(IpoRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def bulk_upsert(self, filings: list[IpoFiling]) -> int:
        if not filings:
            return 0

        # NSE's own feeds can list the same symbol more than once (e.g. a
        # company appearing in both the upcoming and past-issues responses,
        # or the past-issues feed itself repeating a symbol across statuses)
        # - Postgres rejects a single INSERT..ON CONFLICT DO UPDATE batch
        # that targets the same conflict key twice, so dedupe first, keeping
        # the last (most complete/most recent) occurrence per symbol.
        deduped = {f.symbol: f for f in filings}
        filings = list(deduped.values())

        rows = [
            {
                "symbol": f.symbol,
                "company_name": f.company_name,
                "status": f.status,
                "price_range": f.price_range,
                "issue_size": f.issue_size,
                "issue_start_date": f.issue_start_date,
                "issue_end_date": f.issue_end_date,
                "listing_date": f.listing_date,
                "series": f.series,
            }
            for f in filings
        ]

        upserted = 0
        for i in range(0, len(rows), _UPSERT_BATCH_SIZE):
            batch = rows[i : i + _UPSERT_BATCH_SIZE]
            stmt = pg_insert(IpoFilingModel).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol"],
                set_={
                    "company_name": stmt.excluded.company_name,
                    "status": stmt.excluded.status,
                    "price_range": stmt.excluded.price_range,
                    "issue_size": stmt.excluded.issue_size,
                    "issue_start_date": stmt.excluded.issue_start_date,
                    "issue_end_date": stmt.excluded.issue_end_date,
                    "listing_date": stmt.excluded.listing_date,
                    "series": stmt.excluded.series,
                },
            )
            await self._session.execute(stmt)
            upserted += len(batch)

        await self._session.commit()
        return upserted

    async def list_all(self, status: str | None, limit: int, offset: int) -> list[IpoFiling]:
        stmt = select(IpoFilingModel).order_by(IpoFilingModel.updated_at.desc())
        if status is not None:
            stmt = stmt.where(IpoFilingModel.status == status.strip().upper())
        stmt = stmt.offset(offset).limit(limit)

        result = await self._session.execute(stmt)
        return [_to_entity(row) for row in result.scalars().all()]
