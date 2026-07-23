from app.domain.ports import IpoRepositoryPort, StockDataProviderPort
from app.schemas.ipo import IpoFilingOut


def _to_schema(filings) -> list[IpoFilingOut]:
    return [
        IpoFilingOut(
            symbol=f.symbol,
            company_name=f.company_name,
            status=f.status,
            price_range=f.price_range,
            issue_size=f.issue_size,
            issue_start_date=f.issue_start_date,
            issue_end_date=f.issue_end_date,
            listing_date=f.listing_date,
            series=f.series,
        )
        for f in filings
    ]


class IpoService:
    def __init__(self, provider: StockDataProviderPort, repository: IpoRepositoryPort):
        self._provider = provider
        self._repository = repository

    async def sync(self) -> int:
        """Best-effort: raises ProviderUnavailableError only if both source
        endpoints failed (the scheduled job decides how to log it) - the API
        always serves whatever was last synced successfully.
        """
        filings = await self._provider.fetch_ipo_filings()
        if not filings:
            return 0
        return await self._repository.bulk_upsert(filings)

    async def list(self, status: str | None, limit: int, offset: int) -> list[IpoFilingOut]:
        filings = await self._repository.list_all(status, limit, offset)
        return _to_schema(filings)
